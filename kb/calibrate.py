"""阈值标定：用评测 query 集扫一遍阈值网格，选一个"召回够高、误召够低"的值。

评测集格式（每行一条 JSON，人工筛过的定稿文件）：
    {"id": "Q001", "question": "高血压每天吃盐上限是多少？",
     "type": "relevant", "expected_chunk_ids": ["kb::cardio::a1b2c3d4e5f6"]}
    {"id": "N001", "question": "今天天气怎么样", "type": "irrelevant"}

命中的判定按 gold 的精细度自动选：`expected_chunk_ids`（最准）→
`expected_sections` → `expected_doc_ids`（最粗）。一个 doc_id 往往对应一整个科室，
只有 doc 级 gold 时"命中同科室任意一条"就算召回，测不出细粒度差异，所以尽量给
`expected_chunk_ids`。

用法：
    .venv/bin/python -m kb.calibrate --queries kb/eval/queries.jsonl
    .venv/bin/python -m kb.calibrate --queries ... --grid 0.30,0.35,0.40,0.45,0.50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kb.config import get_config
from kb.search import KbIndexMissingError, retrieve_debug

DEFAULT_GRID = "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60"


def load_queries(path: Path) -> list[dict]:
    items: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {lineno} 行不是合法 JSON：{exc}") from exc
    if not items:
        raise ValueError(f"{path} 里没有可用的评测 query")
    return items


def evaluate(items: list[dict], *, k: int, threshold: float | None) -> dict:
    config = get_config()
    # threshold=None 表示「不过滤」这一行基线。注意不能直接把 None 传下去：
    # kb.search 里 None 的语义是「用配置里的阈值」，那样基线行其实是当前阈值的结果。
    effective = -1.0 if threshold is None else threshold
    relevant_total = relevant_hit = 0
    irrelevant_total = irrelevant_false = 0
    returned_sum = 0
    misses: list[str] = []
    false_hits: list[str] = []

    for item in items:
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        hits = retrieve_debug(
            question, k=k, strategy="hybrid", threshold=effective, config=config
        )
        returned_sum += len(hits)
        kind = str(item.get("type") or "relevant")
        if kind == "irrelevant":
            irrelevant_total += 1
            if hits:
                irrelevant_false += 1
                false_hits.append(str(item.get("id")))
            continue
        relevant_total += 1
        chunks = {str(c) for c in (item.get("expected_chunk_ids") or [])}
        sections = {str(s) for s in (item.get("expected_sections") or [])}
        docs = {str(d) for d in (item.get("expected_doc_ids") or [])}
        if chunks:
            matched = any(hit["chunk_id"] in chunks for hit in hits)
        elif sections:
            matched = any(hit["section"] in sections for hit in hits)
        elif docs:
            matched = any(hit["doc_id"] in docs for hit in hits)
        else:
            continue
        if matched:
            relevant_hit += 1
        else:
            misses.append(str(item.get("id")))

    return {
        "threshold": threshold,
        "recall_at_k": round(relevant_hit / relevant_total, 4) if relevant_total else None,
        "false_hit_rate": round(irrelevant_false / irrelevant_total, 4) if irrelevant_total else None,
        "avg_returned": round(returned_sum / max(len(items), 1), 2),
        "n_relevant": relevant_total,
        "n_irrelevant": irrelevant_total,
        "missed_ids": misses,
        "false_hit_ids": false_hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="检索阈值标定")
    parser.add_argument("--queries", type=Path, default=Path("kb/eval/queries.jsonl"))
    parser.add_argument("--k", type=int, default=get_config().top_k)
    parser.add_argument("--grid", default=DEFAULT_GRID, help="逗号分隔的阈值候选")
    parser.add_argument("--max-false-hit", type=float, default=0.10, help="可接受的误召率上限")
    parser.add_argument("--out", type=Path, default=None, help="把逐阈值结果写成 JSON")
    args = parser.parse_args()

    if not args.queries.is_file():
        print(
            f"[calibrate] 找不到评测集 {args.queries}。\n"
            "先生成候选：.venv/bin/python -m kb.eval_set，人工筛过后存成 queries.jsonl"
        )
        return 2

    items = load_queries(args.queries)
    thresholds: list[float | None] = [None] + [float(value) for value in args.grid.split(",") if value.strip()]
    rows: list[dict] = []
    try:
        for threshold in thresholds:
            rows.append(evaluate(items, k=args.k, threshold=threshold))
    except KbIndexMissingError as exc:
        print(f"[calibrate] 无法检索：{exc}")
        return 2

    print(f"评测集：{args.queries}（relevant {rows[0]['n_relevant']} 条 / irrelevant {rows[0]['n_irrelevant']} 条）")
    print(f"{'阈值':>6}{'Recall@' + str(args.k):>12}{'误召率':>10}{'平均条数':>10}")
    for row in rows:
        label = "不过滤" if row["threshold"] is None else f"{row['threshold']:.2f}"
        recall = "—" if row["recall_at_k"] is None else f"{row['recall_at_k']:.4f}"
        false_hit = "—" if row["false_hit_rate"] is None else f"{row['false_hit_rate']:.4f}"
        print(f"{label:>6}{recall:>12}{false_hit:>10}{row['avg_returned']:>10.2f}")

    candidates = [
        row
        for row in rows
        if row["threshold"] is not None
        and (row["false_hit_rate"] is None or row["false_hit_rate"] <= args.max_false_hit)
    ]
    if not candidates:
        print(f"\n[calibrate] 没有阈值能满足误召率 ≤ {args.max_false_hit}，请放宽上限或补语料。")
    else:
        best = max(candidates, key=lambda row: (row["recall_at_k"] or 0.0, row["threshold"]))
        print(
            f"\n[calibrate] 建议 KB_SCORE_THRESHOLD={best['threshold']:.2f}"
            f"（Recall@{args.k}={best['recall_at_k']}，误召率={best['false_hit_rate']}）"
        )
        print(
            "           口径：在误召率达标的前提下取召回最高者；并列时取阈值更高（更保守）的那个。\n"
            "           想让模型多拿一点上下文，可以手动下调，但下调后要重跑本脚本确认误召率没变差。"
        )
        if best["missed_ids"]:
            print(f"           未命中：{', '.join(best['missed_ids'])}")
        if best["false_hit_ids"]:
            print(f"           误召：{', '.join(best['false_hit_ids'])}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[calibrate] 明细写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
