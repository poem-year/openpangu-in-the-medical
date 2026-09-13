# -*- coding: utf-8 -*-
"""把多次 run（例如慢思考 / 快思考两种模式）放在一张表里对比。

用法：
    python eval/compare.py --run-ids L0-slow-b32 L0-fast-b32
    # 结果同时写到第二个 run 目录的 compare.md（也可用 --out 指定）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from progress import pad  # noqa: E402


def load(run_dir: str) -> dict | None:
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def build_table(runs: list[tuple[str, dict]]) -> str:
    head = [name for name, _ in runs]
    lines = ["| 指标 | " + " | ".join(head) + " |",
             "| --- | " + " | ".join("---" for _ in head) + " |"]

    def row(label: str, getter) -> None:
        lines.append(f"| {label} | " + " | ".join(getter(s) for _, s in runs) + " |")

    row("题目数", lambda s: str(s["overall"]["n"]))
    row("可自动判分", lambda s: str(s["overall"]["scored"]))
    row("严格准确率", lambda s: pct(s.get("headline", {}).get("accuracy")))
    row("平均得分（含部分分）", lambda s: pct(s["overall"].get("score_mean")))
    row("安全合规率", lambda s: pct(s.get("headline", {}).get("safety_compliance")))
    row("引用可追溯率", lambda s: pct(s.get("headline", {}).get("citation_traceability")))
    lines.append("")
    lines.append("| 维度 | " + " | ".join(head) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in head) + " |")
    dims = sorted({d for _, s in runs for d in (s.get("dimensions") or {})})
    for dim in dims:
        cells = []
        for _, s in runs:
            agg = (s.get("dimensions") or {}).get(dim)
            cells.append("—" if not agg else pct(agg.get("accuracy")))
        lines.append(f"| {dim} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_terminal(runs: list[tuple[str, dict]]) -> str:
    head = [name for name, _ in runs]
    width = max(18, *(len(h) + 2 for h in head))
    out = ["═" * (24 + width * len(head)), " 模式对比", "═" * (24 + width * len(head))]

    def line(label: str, values: list[str]) -> None:
        out.append(f" {pad(label, 22)}" + "".join(pad(v, width, "right") for v in values))

    line("题目数", [str(s["overall"]["n"]) for _, s in runs])
    line("可自动判分", [str(s["overall"]["scored"]) for _, s in runs])
    line("严格准确率", [pct(s.get("headline", {}).get("accuracy")) for _, s in runs])
    line("平均得分（含部分分）", [pct(s["overall"].get("score_mean")) for _, s in runs])
    line("安全合规率", [pct(s.get("headline", {}).get("safety_compliance")) for _, s in runs])
    line("引用可追溯率", [pct(s.get("headline", {}).get("citation_traceability")) for _, s in runs])
    out.append("-" * (24 + width * len(head)))
    dims = sorted({d for _, s in runs for d in (s.get("dimensions") or {})})
    for dim in dims:
        values = []
        for _, s in runs:
            agg = (s.get("dimensions") or {}).get(dim)
            values.append("—" if not agg else pct(agg.get("accuracy")))
        line(dim, values)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--runs-root", default="/data/openpangu/eval/runs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runs = []
    for rid in args.run_ids:
        data = load(os.path.join(args.runs_root, rid))
        if data:
            runs.append((rid, data))
        else:
            print(f"（跳过 {rid}：没有 summary.json）", file=sys.stderr)
    if len(runs) < 2:
        print("至少需要两个跑完的 run 才能对比", file=sys.stderr)
        sys.exit(1)

    print(render_terminal(runs))
    out_path = args.out or os.path.join(args.runs_root, args.run_ids[-1], "compare.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# 模式对比\n\n" + build_table(runs) + "\n")
    print(f"\n对比表已写入 {out_path}")


if __name__ == "__main__":
    main()
