"""检索评测集生成：从语料反向生成候选 query，供人工筛选后定稿。

产物是**候选**，不是定稿：每条带 `reviewed: false`，人工筛过后把
`reviewed` 改成 true 并整理成 `kb/eval/queries.jsonl`，再用 `kb.calibrate` 标定阈值。

用法：
    .venv/bin/python -m kb.eval_set                       # 每个小节生成 1 条候选 query
    .venv/bin/python -m kb.eval_set --per-section 2 --irrelevant 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kb.config import get_config
from kb.search import KbIndexMissingError, load_index

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from judge import DEFAULT_MODEL, read_base_url, read_token  # noqa: E402

QUERY_SYSTEM = (
    "你是医学知识库的评测集作者。用户会给你一段医学资料，请写出患者真实会问、"
    "且**仅凭这段资料就能回答**的问题。问题要口语化、像患者在问诊平台上的提问，"
    "不要出现「根据上文」「这段材料」这类字眼，也不要直接抄资料里的句子。"
    "只输出 JSON：{\"questions\": [\"问题1\", \"问题2\"]}"
)

IRRELEVANT_SYSTEM = (
    "请生成与医学知识无关的日常提问（如天气、交通、编程、生活杂物），"
    "用于测试检索系统在无关问题上的误召。只输出 JSON：{\"questions\": [\"...\", \"...\"]}"
)


def call_model(system: str, user: str, *, token: str, base_url: str, model: str, timeout: int = 120) -> list[str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:200]
        raise RuntimeError(f"模型调用失败 HTTP {exc.code}：{detail}") from exc
    content = body["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise RuntimeError(f"模型输出不是 JSON：{content[:120]}")
    questions = json.loads(match.group(0)).get("questions") or []
    return [str(item).strip() for item in questions if str(item).strip()]


def sample_sections(
    index,
    per_section: int,
    *,
    max_per_doc: int | None = None,
    limit: int | None = None,
    seed: int = 20260914,
) -> list[dict]:
    """按 (doc_id, section) 取样，每个小节取 first-N 条正文代表。

    取样要**跨文档**才标得准：索引里 125 个文档、4.5 万个小节，前 40 个小节全落在
    第 1 卷百科里，标出来的阈值只反映「百科 vs 百科」。所以这里先按固定种子打散，
    再限制单文档最多贡献几个小节，最后等距抽到 limit 条——同一 seed 结果完全可复现。
    """
    buckets: dict[tuple[str, str], list] = {}
    for record in index.records:
        buckets.setdefault((record.doc_id, record.section), []).append(record)

    items = list(buckets.items())
    random.Random(seed).shuffle(items)

    samples: list[dict] = []
    per_doc: dict[str, int] = {}
    for (doc_id, section), records in items:
        if max_per_doc is not None and per_doc.get(doc_id, 0) >= max_per_doc:
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        for record in records[:per_section]:
            samples.append({"doc_id": doc_id, "section": section, "record": record})

    if limit is not None and len(samples) > limit:
        step = len(samples) / limit
        samples = [samples[int(i * step)] for i in range(limit)]
    return samples


def main() -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description="生成检索评测集候选问题")
    parser.add_argument("--out", type=Path, default=Path("kb/eval/queries_candidates.jsonl"))
    parser.add_argument("--per-section", type=int, default=1, help="每个小节取几条切块当素材")
    parser.add_argument("--irrelevant", type=int, default=8, help="额外生成多少条无关问题")
    parser.add_argument("--limit", type=int, default=None, help="取样上限（跨文档等距抽）")
    parser.add_argument("--max-per-doc", type=int, default=None, help="单文档最多贡献几个小节")
    parser.add_argument("--seed", type=int, default=20260914, help="取样随机种子（固定则可复现）")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    try:
        index = load_index(config)
    except KbIndexMissingError as exc:
        print(f"[eval_set] 无法读取索引：{exc}")
        return 2

    token, base_url = read_token(), read_base_url()
    samples = sample_sections(
        index,
        args.per_section,
        max_per_doc=args.max_per_doc,
        limit=args.limit,
        seed=args.seed,
    )
    print(f"[eval_set] 从 {index.size} 条切块中取样 {len(samples)} 条作为素材，模型 {args.model}")

    def build_one(sample: dict) -> dict | None:
        record = sample["record"]
        prompt = f"【资料】\n{record.text}\n\n请生成 1~2 个问题。"
        try:
            questions = call_model(
                QUERY_SYSTEM, prompt, token=token, base_url=base_url, model=args.model
            )
        except Exception as exc:  # noqa: BLE001 - 单条失败不影响整体
            print(f"[eval_set][跳过] {record.chunk_id}：{type(exc).__name__}: {exc}")
            return None
        return {
            "questions": questions,
            "doc_id": sample["doc_id"],
            "section": sample["section"],
            "chunk_id": record.chunk_id,
        }

    rows: list[dict] = []
    counter = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for result in pool.map(build_one, samples):
            if not result:
                continue
            for question in result["questions"]:
                counter += 1
                rows.append(
                    {
                        "id": f"Q{counter:04d}",
                        "question": question,
                        "type": "relevant",
                        # chunk 级 gold：问题就是从这块切块写出来的，所以「检索有没有把
                        # 它找回来」是可测的。doc 级不能当 gold——一个 doc_id 就是一整卷
                        # 百科（约 1700 块），命中任意一块都算召回，等于不测排序。
                        "expected_chunk_ids": [result["chunk_id"]],
                        "expected_doc_ids": [result["doc_id"]],
                        "expected_sections": [result["section"]],
                        "reviewed": False,
                    }
                )

    if args.irrelevant:
        try:
            extra = call_model(
                IRRELEVANT_SYSTEM,
                f"请生成 {args.irrelevant} 条与医学无关的问题。",
                token=token,
                base_url=base_url,
                model=args.model,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[eval_set][跳过] 无关问题生成失败：{type(exc).__name__}: {exc}")
            extra = []
        for question in extra[: args.irrelevant]:
            counter += 1
            rows.append(
                {
                    "id": f"N{counter:04d}",
                    "question": question,
                    "type": "irrelevant",
                    "expected_doc_ids": [],
                    "reviewed": False,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    relevant = sum(1 for row in rows if row["type"] == "relevant")
    print(
        f"[eval_set] 生成候选 {len(rows)} 条（相关 {relevant} / 无关 {len(rows) - relevant}）"
        f"→ {args.out}\n"
        "           请人工筛一遍：删掉不该出现的、把 reviewed 改成 true，"
        "另存为 kb/eval/queries.jsonl 后跑 kb.calibrate。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
