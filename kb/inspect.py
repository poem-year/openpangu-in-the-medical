"""检索调试 CLI：看 top-k、各阶段分数与来源分布。

用法：
    .venv/bin/python -m kb.inspect "高血压诊断标准是什么"
    .venv/bin/python -m kb.inspect "高血压" --strategy vector --threshold 0.0 -k 3
"""

from __future__ import annotations

import argparse

from kb.config import STRATEGY_HYBRID, STRATEGY_VECTOR, get_config
from kb.search import KbIndexMissingError, load_index, retrieve_debug


def _preview(text: str, width: int = 80) -> str:
    single = " ".join(text.split())
    return single if len(single) <= width else single[: width - 1] + "…"


def main() -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description="检索调试：看 top-k 与分数分布")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("-k", type=int, default=config.top_k)
    parser.add_argument("--strategy", choices=[STRATEGY_VECTOR, STRATEGY_HYBRID], default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="向量余弦下限；不给则 hybrid 用配置值、vector 不过滤",
    )
    parser.add_argument("--full-text", action="store_true", help="打印完整正文而不是摘要")
    args = parser.parse_args()

    try:
        index = load_index(config)
    except KbIndexMissingError as exc:
        print(f"[inspect] 无法检索：{exc}")
        return 2

    strategy = args.strategy or config.strategy
    hits = retrieve_debug(
        args.query, k=args.k, strategy=strategy, threshold=args.threshold, config=config
    )

    manifest = index.manifest or {}
    print(
        f"索引：{index.size} 条切块 / {manifest.get('n_docs', '?')} 个文档 / "
        f"{index.dim} 维 / 模型 {manifest.get('embed_model', '?')}"
    )
    print(f"查询：{args.query!r}　策略：{strategy}　top-k：{args.k}")
    if not hits:
        print("结果：无（没有候选过阈值）")
        return 0

    print()
    header = f"{'#':<3}{'余弦':>8}{'向量':>6}{'BM25':>6}{'融合分':>9}  {'来源 / 章节':<28}chunk_id"
    print(header)
    print("-" * len(header))
    for rank, hit in enumerate(hits, start=1):
        origin = f"{hit['source']} / {hit['section']}"[:27]
        print(
            f"{rank:<3}{hit['cosine']:>8.4f}{str(hit['vector_rank'] or '-'):>6}"
            f"{str(hit['bm25_rank'] or '-'):>6}{hit['score']:>9.4f}  {origin:<28}{hit['chunk_id']}"
        )

    print("\n正文：")
    for rank, hit in enumerate(hits, start=1):
        body = hit["text"] if args.full_text else _preview(hit["text"])
        print(f"  [{rank}] {body}")

    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit["doc_id"]] = counts.get(hit["doc_id"], 0) + 1
    print("\n来源分布：" + "、".join(f"{doc}×{n}" for doc, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
