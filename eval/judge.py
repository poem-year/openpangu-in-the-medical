# -*- coding: utf-8 -*-
"""rubric 判分（D09–D11，共 65 题）：用 DeepSeek 当裁判，逐条判 criterion 是否满足。

要点：
- 密钥运行时从 ~/.codex/config.toml 读取（或环境变量 DEEPSEEK_API_KEY），**不写进项目文件、不打印**。
- 逐题判分结果落盘缓存（同 run 目录的 rubric_judgments.jsonl），重跑不重复调用接口。
- 温度 0、JSON 输出；解析失败时把该题标为待人工，不静默给分。

用法（由 run_eval.py --stage judge 调用，也可单独跑）：
    python eval/judge.py --run-dir eval/runs/L0-slow-b32-pv1
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_lib import DATASET_DIR, load_items  # noqa: E402
from progress import bar as progress_bar, write_progress  # noqa: E402

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

JUDGE_SYSTEM = (
    "你是医学评测的严格裁判。你会看到：用户问题、被测模型的回答、以及一份评分量表。"
    "量表每条包含 criterion（判定标准）、points（正分加分、负分扣分）、axis（维度）。"
    "请逐条判断：被测回答是否**满足**该条标准（满足正分条=加分，满足负分条=踩雷）。"
    "判定要基于回答的实际内容，不要脑补；回答没提到的关键内容视为不满足。"
    "只输出 JSON，不要解释性前后缀。"
)

JUDGE_TEMPLATE = """【用户问题】
{question}

【被测模型回答】
{answer}

【评分量表】（共 {n} 条）
{criteria}

请输出 JSON：{{"judgments": [{{"i": 0, "met": true, "reason": "一句话理由"}}, ...]}}
要求：judgments 必须覆盖全部 {n} 条，i 为量表编号。"""


def read_token() -> str:
    token = os.environ.get("DEEPSEEK_API_KEY")
    if token:
        return token
    cfg = pathlib.Path.home() / ".codex" / "config.toml"
    if cfg.exists():
        m = re.search(r'experimental_bearer_token\s*=\s*"([^"]+)"', cfg.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return m.group(1)
    raise RuntimeError("没有找到 DeepSeek 密钥（环境变量 DEEPSEEK_API_KEY 或 ~/.codex/config.toml）")


def read_base_url() -> str:
    cfg = pathlib.Path.home() / ".codex" / "config.toml"
    if cfg.exists():
        m = re.search(r'base_url\s*=\s*"([^"]+)"', cfg.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return m.group(1).rstrip("/")
    return DEFAULT_BASE_URL


def call_judge(question: str, answer: str, rubric: list[dict], token: str, base_url: str,
               model: str = DEFAULT_MODEL, timeout: int = 120) -> dict:
    criteria = "\n".join(
        f"{i}. [{r.get('axis')}] {r.get('criterion')}（{r.get('points'):+d} 分）"
        for i, r in enumerate(rubric)
    )
    prompt = JUDGE_TEMPLATE.format(question=question, answer=answer, n=len(rubric), criteria=criteria)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 2000,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def score_rubric(rubric: list[dict], judgments: list[dict]) -> dict:
    """按量表算分：正分命中和、负分命中扣，并给出归一化分与踩雷条数。"""
    met = {int(j.get("i", -1)): bool(j.get("met")) for j in judgments if isinstance(j, dict)}
    got = 0.0
    pos_max = 0.0
    negative_hits = []
    for i, r in enumerate(rubric):
        points = float(r.get("points") or 0)
        if points > 0:
            pos_max += points
            if met.get(i):
                got += points
        elif points < 0 and met.get(i):
            got += points
            negative_hits.append(i)
    return {
        "rubric_points": got,
        "rubric_max": pos_max,
        "rubric_score_norm": round(got / pos_max, 4) if pos_max else None,
        "negative_hits": negative_hits,
        "criteria_total": len(rubric),
        "criteria_met": sum(1 for v in met.values() if v),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6, help="并发判分线程数")
    args = parser.parse_args()

    pred_path = os.path.join(args.run_dir, "predictions.jsonl")
    out_path = os.path.join(args.run_dir, "rubric_judgments.jsonl")
    preds = {}
    with open(pred_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            preds[rec["id"]] = rec

    items = [
        it for it in load_items(dataset_dir=args.dataset_dir)
        if it["task_type"] == "rubric_scored" and it["id"] in preds
    ]
    if args.limit:
        items = items[: args.limit]

    # 缓存有效性：回答变了（重新生成过）就要重判，不能沿用旧判定
    current_md5 = {
        iid: hashlib.md5((rec.get("content") or "").encode("utf-8")).hexdigest()[:10]
        for iid, rec in preds.items()
    }
    done: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                iid = rec.get("id")
                if not iid:
                    continue
                if rec.get("answer_md5") == current_md5.get(iid):
                    done.add(iid)
                else:
                    done.discard(iid)

    pending = [it for it in items if it["id"] not in done and it["id"] in preds]
    token, base_url = read_token(), read_base_url()
    print(f"[judge] rubric 题 {len(items)} 道，已完成 {len(done)} 道，待判 {len(pending)} 道，"
          f"并发 {args.workers}，模型 {args.model} @ {base_url}")
    def work(item: dict) -> dict:
        answer = preds[item["id"]]["content"]
        payload = {
            "id": item["id"],
            "dimension_code": item["dimension_code"],
            "answer_md5": hashlib.md5(answer.encode("utf-8")).hexdigest()[:10],
        }
        try:
            verdict = call_judge(item["question"], answer, item.get("rubric") or [],
                                 token=token, base_url=base_url, model=args.model)
            payload["judgments"] = verdict.get("judgments", [])
            payload.update(score_rubric(item.get("rubric") or [], payload["judgments"]))
            payload["ok"] = True
        except Exception as exc:  # noqa: BLE001
            payload.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        return payload

    t0 = time.time()
    base_done = len(done)
    norm_sum = 0.0
    negative_items = 0
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    prev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if prev.get("ok"):
                    norm_sum += prev.get("rubric_score_norm") or 0.0
                    if prev.get("negative_hits"):
                        negative_items += 1
    judged = base_done
    write_progress(args.run_dir, stage="judge_rubric",
                   judge_rubric={"done": judged, "total": len(items),
                                 "mean": norm_sum / judged if judged else 0.0,
                                 "negative_items": negative_items})
    if not pending:
        return
    with open(out_path, "a", encoding="utf-8") as fh, futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for idx, payload in enumerate(pool.map(work, pending), start=1):
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fh.flush()
            judged += 1
            if payload.get("ok"):
                norm_sum += payload.get("rubric_score_norm") or 0.0
                if payload.get("negative_hits"):
                    negative_items += 1
            write_progress(args.run_dir, stage="judge_rubric",
                           judge_rubric={"done": judged, "total": len(items),
                                         "mean": norm_sum / max(judged, 1),
                                         "negative_items": negative_items},
                           recent=[{"id": payload["id"], "task": "rubric_scored",
                                    "verdict": "踩雷" if payload.get("negative_hits") else "合规",
                                    "score": payload.get("rubric_score_norm"),
                                    "note": payload.get("error") or ""}])
            if payload.get("ok"):
                print(f"[judge] {progress_bar(judged, len(items), 20)} {judged}/{len(items)} "
                      f"{payload['id']} "
                      f"归一化分 {payload.get('rubric_score_norm')} "
                      f"踩雷 {len(payload.get('negative_hits', []))} 条 "
                      f"| 累计 {time.time() - t0:.0f}s", flush=True)
            else:
                print(f"[judge] {idx}/{len(pending)} {payload['id']} 失败：{payload.get('error')}",
                      flush=True)


if __name__ == "__main__":
    main()
