# -*- coding: utf-8 -*-
"""把 summary.json 渲染成终端里能直接看的报告（一键脚本最后一步调用）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata


def width(text: str) -> int:
    """中文按 2 列计算，保证终端表格对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text: str, w: int, align: str = "left") -> str:
    text = str(text)
    fill = max(0, w - width(text))
    if align == "right":
        return " " * fill + text
    return text + " " * fill


def pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def render(summary: dict, run_dir: str) -> str:
    lines: list[str] = []
    bar = "═" * 78
    lines.append(bar)
    lines.append(f" openPangu-R-7B-2512 医学诊断能力评测 · {summary.get('run_id', '')}")
    lines.append(f" 层级 {summary.get('layer')} ｜ 思考模式 {summary.get('thinking_mode')} ｜ "
                 f"批大小 {summary.get('batch_size')} ｜ 配置指纹 {summary.get('fingerprint')}")
    lines.append(bar)

    h = summary.get("headline") or {}
    o = summary.get("overall") or {}
    lines.append("")
    lines.append("【四个上报指标】")
    lines.append(f"  准确率（严格口径）      {pct(h.get('accuracy'))}")
    lines.append(f"  平均得分（含部分分）    {pct(o.get('score_mean'))}")
    lines.append(f"  安全合规率（rubric）    {pct(h.get('safety_compliance'))}")
    if h.get("rag_enabled"):
        lines.append(f"  引用可追溯率            {pct(h.get('citation_traceability'))}"
                     f"（{h.get('citation_traceability_note', '')}）")
    else:
        lines.append("  引用可追溯率            0.0%（知识库模块指标；基线无检索、无引用，按定义计 0）")
    lines.append("")
    lines.append(f"  题目数 {o.get('n')}（可自动判分 {o.get('scored')}，待判分 {o.get('pending')}，"
                 f"触顶截断 {o.get('truncated')}）")
    ci = o.get("ci95")
    if ci:
        lines.append(f"  准确率 95% 置信区间     [{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]")

    dims = summary.get("dimensions") or {}
    if dims:
        lines.append("")
        lines.append("【分维度表现】")
        cols = [("维度", 6), ("题量", 5), ("可判分", 7), ("正确", 5),
                ("准确率", 8), ("平均分", 8), ("待判分", 7), ("截断", 5)]
        header = "  " + " ".join(pad(name, w) for name, w in cols)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for dim, agg in dims.items():
            row = [
                pad(dim, 6),
                pad(agg.get("n"), 5, "right"),
                pad(agg.get("scored"), 7, "right"),
                pad(agg.get("correct"), 5, "right"),
                pad(pct(agg.get("accuracy")), 8, "right"),
                pad(pct(agg.get("score_mean")), 8, "right"),
                pad(agg.get("pending"), 7, "right"),
                pad(agg.get("truncated", 0), 5, "right"),
            ]
            lines.append("  " + " ".join(row))

    rubric = summary.get("rubric") or {}
    if rubric:
        lines.append("")
        lines.append("【rubric 判分（D09–D11，按 criterion 逐条判）】")
        lines.append("  " + " ".join([pad("维度", 6), pad("题量", 5, "right"),
                                       pad("平均归一化分", 14, "right"), pad("踩雷题数", 9, "right"),
                                       pad("踩雷率", 8, "right")]))
        lines.append("  " + "-" * 46)
        for dim, agg in rubric.items():
            lines.append("  " + " ".join([
                pad(dim, 6),
                pad(agg.get("n"), 5, "right"),
                pad(pct(agg.get("mean_rubric_score")), 14, "right"),
                pad(agg.get("negative_hit_items"), 9, "right"),
                pad(pct(agg.get("negative_hit_rate")), 8, "right"),
            ]))

    lines.append("")
    lines.append("【产物】")
    for name in ("report.html", "summary.md", "summary.json", "scores.jsonl", "predictions.jsonl",
                 "answer_judgments.jsonl", "rubric_judgments.jsonl", "skipped.jsonl"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            lines.append(f"  {pad(name, 24)} {size:8.1f} KB   {path}")
    lines.append(bar)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--json", action="store_true", help="直接输出 summary.json")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(render(summary, args.run_dir))


def write_html(summary: dict, path: str) -> None:
    """生成一个不自带依赖的 HTML 报告（浏览器/VS Code 预览都能直接打开）。"""

    def pct(v):
        return "—" if v is None else f"{v * 100:.1f}%"

    def bar_cell(v, color="#2f6feb"):
        if v is None:
            return "—"
        width = max(0.0, min(1.0, float(v))) * 100
        return (f'<div class="bar"><span style="width:{width:.1f}%;background:{color}"></span></div>'
                f'<span class="num">{pct(v)}</span>')

    h = summary.get("headline") or {}
    o = summary.get("overall") or {}
    dims = summary.get("dimensions") or {}
    rubric = summary.get("rubric") or {}

    cards = [
        ("严格准确率", pct(h.get("accuracy"))),
        ("平均得分（含部分分）", pct(o.get("score_mean"))),
        ("安全合规率", pct(h.get("safety_compliance"))),
        ("引用可追溯率（知识库指标）",
         "0.0%" if not h.get("rag_enabled") else pct(h.get("citation_traceability"))),
        ("题目数", f"{o.get('n', 0)}"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards
    )
    rows = "".join(
        f"<tr><td>{dim}</td><td class='num'>{agg.get('n')}</td><td class='num'>{agg.get('scored')}</td>"
        f"<td class='num'>{agg.get('correct')}</td><td>{bar_cell(agg.get('accuracy'))}</td>"
        f"<td>{bar_cell(agg.get('score_mean'), '#8a5cf6')}</td>"
        f"<td class='num'>{agg.get('pending')}</td><td class='num'>{agg.get('truncated', 0)}</td></tr>"
        for dim, agg in dims.items()
    )
    rubric_rows = "".join(
        f"<tr><td>{dim}</td><td class='num'>{agg.get('n')}</td>"
        f"<td>{bar_cell(agg.get('mean_rubric_score'), '#0f9d58')}</td>"
        f"<td class='num'>{agg.get('negative_hit_items')}</td>"
        f"<td>{bar_cell(agg.get('negative_hit_rate'), '#d93025')}</td></tr>"
        for dim, agg in rubric.items()
    )
    rubric_section = (
        "<h2>rubric 判分（D09–D11）</h2><table><thead><tr>"
        "<th>维度</th><th>题量</th><th>平均归一化分</th><th>踩雷题数</th><th>踩雷率</th>"
        f"</tr></thead><tbody>{rubric_rows}</tbody></table>"
    ) if rubric_rows else ""

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>评测报告 · {summary.get('run_id', '')}</title>
<style>
 body {{ font-family: -apple-system, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
        margin: 32px auto; max-width: 1000px; color: #1f2328; }}
 h1 {{ font-size: 22px; margin-bottom: 4px; }}
 .meta {{ color: #57606a; font-size: 13px; margin-bottom: 20px; }}
 .cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }}
 .card {{ border: 1px solid #d0d7de; border-radius: 10px; padding: 12px 16px; min-width: 150px; }}
 .card .k {{ font-size: 12px; color: #57606a; }}
 .card .v {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ border-bottom: 1px solid #eaeef2; padding: 7px 8px; text-align: left; }}
 th {{ background: #f6f8fa; font-weight: 600; }}
 .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 .bar {{ display: inline-block; width: 130px; height: 10px; background: #eaeef2;
         border-radius: 5px; overflow: hidden; vertical-align: middle; margin-right: 8px; }}
 .bar span {{ display: block; height: 100%; }}
</style></head><body>
<h1>openPangu-R-7B-2512 · 医学诊断能力评测</h1>
<div class="meta">运行 {summary.get('run_id')} ｜ 层级 {summary.get('layer')} ｜
 思考模式 {summary.get('thinking_mode')} ｜ 批大小 {summary.get('batch_size')} ｜
 配置指纹 {summary.get('fingerprint')}</div>
<div class="cards">{card_html}</div>
<h2>分维度表现</h2>
<table><thead><tr><th>维度</th><th>题量</th><th>可判分</th><th>正确</th><th>严格准确率</th>
<th>平均得分</th><th>待判分</th><th>截断</th></tr></thead><tbody>{rows}</tbody></table>
{rubric_section}
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


if __name__ == "__main__":
    sys.exit(main())
