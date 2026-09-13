# -*- coding: utf-8 -*-
"""开放题 AI 复核判分：规则能判的直接判，判不了的交给大模型判「是否同一个诊断」。

两级判分（快路 + 复核）：
1. 快路（零成本、确定性）：参考答案里的主诊断名（或鉴别诊断条目）完整出现在回答里 → 直接判对。
   选择题、计算题、yes/no 题不走这里，它们由 eval_lib.score_item 精确判。
2. AI 复核：其余情况给 DeepSeek 看「问题 + 材料 + 参考答案 + 参考要点 + 模型回答」，
   输出结构化判定：verdict（correct / partial / wrong）、score（0–1）、same_disease、reason，
   以及回答中是否存在与参考答案冲突或明显编造的内容（供幻觉率参考）。

判分器看不到模型身份，温度 0，结果按「题号 + 回答哈希」缓存，重跑不重复调用。
密钥运行时从 ~/.codex/config.toml 或环境变量读取，不落项目文件、不打印。
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_lib import DATASET_DIR, extract_ddx_list, extract_diagnosis, load_items, normalize  # noqa: E402
from judge import DEFAULT_MODEL, read_base_url, read_token  # noqa: E402
from progress import bar as progress_bar, write_progress  # noqa: E402

# 需要 AI 复核的题型（标准答案不唯一，规则判不准）
OPEN_TASKS = {
    "open_diagnosis",
    "dialogue_diagnosis",
    "open_ddx",
    "open_ranked_ddx",
    "open_workup",
}

JUDGE_SYSTEM = (
    "你是医学诊断评测的严格裁判。你会看到一道题、参考答案（含参考要点）和被评测模型的回答。"
    "请判断模型的回答在医学语义上是否与参考答案一致，重点看诊断结论本身，"
    "不要因为措辞、顺序、详略不同就判错，也不要因为提到相关疾病就判对。判分标准："
    "correct（score=1）＝核心诊断或结论与参考答案一致，或为公认同义说法（含英文名、缩写、常见别名）；"
    "partial（score=0.5）＝方向对但不够精确，例如只答到上位诊断、漏掉关键限定词、"
    "或只覆盖了合并诊断的一部分、鉴别诊断只列出部分条目；"
    "wrong（score=0）＝核心结论与参考答案不同，或答非所问。"
    "另外单独标记回答里是否存在与参考答案冲突的编造内容（hallucination=true/false）。只输出 JSON。"
)

TEMPLATE = """【题目】
{question}

【材料（如有）】
{context}

【参考答案】
{reference}

【参考要点】
{key_points}

【被评测模型的回答】
{answer}

请输出 JSON：
{{"verdict": "correct|partial|wrong", "score": 1.0, "same_disease": true,
"missing": ["回答漏掉的关键点"], "hallucination": false,
"reason": "一句话说明判定依据"}}"""


def rule_fast_path(item: dict, answer: str) -> dict | None:
    """快路：参考答案的核心结论完整出现在回答里，直接判对。"""
    ans = normalize(answer)
    if not ans:
        return None
    task = item["task_type"]
    if task in ("open_diagnosis", "dialogue_diagnosis"):
        want = extract_diagnosis(item)
        if want and normalize(want) and normalize(want) in ans:
            return {"verdict": "correct", "score": 1.0, "same_disease": True,
                    "reason": f"回答完整包含参考诊断「{want}」", "source": "rule"}
    if task in ("open_ddx", "open_ranked_ddx"):
        expects = extract_ddx_list(item)
        hits = [e for e in expects if normalize(e) and normalize(e) in ans]
        if expects and len(hits) == len(expects):
            return {"verdict": "correct", "score": 1.0, "same_disease": True,
                    "reason": f"参考鉴别诊断 {len(expects)} 条全部命中", "source": "rule"}
    return None


def build_prompt(item: dict, answer: str) -> str:
    key_points = item.get("key_points") or []
    if not key_points and item.get("reference_answer"):
        key_points = [item["reference_answer"]]
    return TEMPLATE.format(
        question=item.get("question", ""),
        context=(item.get("context") or "")[:3000] or "（无）",
        reference=item.get("reference_answer") or "（无）",
        key_points="\n".join(f"- {k}" for k in key_points[:8]) or "（无）",
        answer=(answer or "")[:4000],
    )


def judge_one(item: dict, answer: str, token: str, base_url: str, model: str) -> dict:
    fast = rule_fast_path(item, answer)
    if fast:
        return fast
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_prompt(item, answer)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 800,
    }
    import urllib.request

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    verdict = json.loads(data["choices"][0]["message"]["content"])
    verdict["source"] = "ai"
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=6, help="并发调用数")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-types", nargs="*", default=sorted(OPEN_TASKS))
    args = parser.parse_args()

    preds = {}
    with open(os.path.join(args.run_dir, "predictions.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            preds[rec["id"]] = rec

    items = [
        it for it in load_items(dataset_dir=args.dataset_dir)
        if it["task_type"] in set(args.task_types) and it["id"] in preds
    ]
    if args.limit:
        items = items[: args.limit]

    out_path = os.path.join(args.run_dir, "answer_judgments.jsonl")
    # 缓存有效性：回答变了（例如重新生成过）就必须重判，不能沿用旧判定
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
    pending = [it for it in items if it["id"] not in done]
    print(f"[ai-judge] 开放题 {len(items)} 道，已完成 {len(done)} 道，待判 {len(pending)} 道，"
          f"并发 {args.workers}，模型 {args.model}")
    if not pending:
        return

    token, base_url = read_token(), read_base_url()
    t0 = time.time()
    tally = {"correct": 0, "partial": 0, "wrong": 0, "rule": 0}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    prev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if prev.get("source") == "rule":
                    tally["rule"] += 1
                if prev.get("verdict") in tally:
                    tally[prev["verdict"]] += 1
    total_all = len(items)
    completed = len(done)
    write_progress(args.run_dir, stage="judge_open",
                   judge_open={"done": completed, "total": total_all, **tally})
    with open(out_path, "a", encoding="utf-8") as fh, \
            futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(judge_one, it, preds[it["id"]]["content"], token, base_url, args.model): it
            for it in pending
        }
        for idx, job in enumerate(futures.as_completed(jobs), start=1):
            item = jobs[job]
            answer = preds[item["id"]]["content"] or ""
            try:
                verdict = job.result()
                ok, err = True, None
            except Exception as exc:  # noqa: BLE001
                verdict, ok, err = {}, False, f"{type(exc).__name__}: {exc}"
            rec = {
                "id": item["id"],
                "dimension_code": item["dimension_code"],
                "task_type": item["task_type"],
                "answer_md5": hashlib.md5(answer.encode("utf-8")).hexdigest()[:10],
                "ok": ok,
                "error": err,
                **verdict,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            completed += 1
            if rec.get("source") == "rule":
                tally["rule"] += 1
            if rec.get("verdict") in tally:
                tally[rec["verdict"]] += 1
            recent = [{"id": rec["id"], "task": rec["task_type"], "verdict": rec.get("verdict", ""),
                       "score": rec.get("score"), "note": (rec.get("reason") or rec.get("error") or "")[:40]}]
            write_progress(args.run_dir, stage="judge_open",
                           judge_open={"done": completed, "total": total_all, **tally},
                           recent=recent)
            print(f"[ai-judge] {progress_bar(completed, total_all, 20)} {completed}/{total_all} "
                  f"{item['id']} {rec.get('verdict')} {rec.get('score')} ({rec.get('source')}) "
                  f"对{tally['correct']}/部分{tally['partial']}/错{tally['wrong']} | "
                  f"{(rec.get('reason') or rec.get('error') or '')[:40]} | {time.time() - t0:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
