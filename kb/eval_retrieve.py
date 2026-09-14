"""评测预检索阶段：为 L2/L3 生成上下文，供 run_eval.py 的生成阶段消费。

为什么单独一个脚本：`eval/run_eval.py` 顶层 import 了 `pangu_infer`，只能在
`.venv-pangu` 里跑；本脚本只用检索依赖，跑在 `.venv` 里，两边互不污染。

用法：
    .venv/bin/python -m kb.eval_retrieve --input items.jsonl --out contexts.jsonl --layer L2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kb.config import STRATEGY_HYBRID, STRATEGY_VECTOR, get_config
from kb.search import KbIndexMissingError, retrieve_debug

LAYER_STRATEGY = {"L2": STRATEGY_VECTOR, "L3": STRATEGY_HYBRID, "L5": STRATEGY_HYBRID}


def read_items(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def main() -> int:
    config = get_config()
    parser = argparse.ArgumentParser(description="评测预检索：写 contexts.jsonl")
    parser.add_argument("--input", type=Path, required=True, help="待检索题目 jsonl（含 id/question）")
    parser.add_argument("--out", type=Path, required=True, help="输出 contexts.jsonl")
    parser.add_argument("--layer", default="L2", help="评测层级（L2 走纯向量，L3/L5 走混合检索）")
    parser.add_argument("-k", type=int, default=config.top_k)
    parser.add_argument("--strategy", default=None, help="覆盖层级默认策略：vector / hybrid")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖阈值（不给则 L2 不过滤、L3 用配置值）")
    args = parser.parse_args()

    layer = args.layer.upper()
    strategy = args.strategy or LAYER_STRATEGY.get(layer, STRATEGY_HYBRID)
    items = read_items(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    total_hits = empty = 0
    print(f"[retrieve] 层级 {layer} → 策略 {strategy}，共 {len(items)} 题，k={args.k}")
    with open(args.out, "w", encoding="utf-8") as fh:
        for index, item in enumerate(items, start=1):
            question = str(item.get("question") or "").strip()
            try:
                hits = (
                    retrieve_debug(
                        question, k=args.k, strategy=strategy, threshold=args.threshold, config=config
                    )
                    if question
                    else []
                )
            except KbIndexMissingError as exc:
                print(f"[retrieve] 无法检索：{exc}")
                return 2
            total_hits += len(hits)
            empty += 0 if hits else 1
            fh.write(
                json.dumps(
                    {
                        "id": item.get("id"),
                        "question": question,
                        "layer": layer,
                        "strategy": strategy,
                        "threshold": args.threshold,
                        "evidence": hits,
                        "retrieved_chunk_ids": [hit["chunk_id"] for hit in hits],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if index % 20 == 0 or index == len(items):
                elapsed = time.time() - started
                print(
                    f"[retrieve] {index}/{len(items)}　平均命中 {total_hits / index:.1f} 条　"
                    f"空结果 {empty} 题　已用 {elapsed:.0f}s"
                )

    elapsed = time.time() - started
    print(
        f"[retrieve] 完成：{len(items)} 题，平均 {total_hits / max(len(items), 1):.2f} 条证据，"
        f"其中 {empty} 题无证据，耗时 {elapsed:.1f}s → {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
