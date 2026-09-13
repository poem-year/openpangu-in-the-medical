# -*- coding: utf-8 -*-
"""评测进度状态：各阶段实时写 progress.json，终端/看板据此渲染进度。

设计：状态文件放在 run 目录下，谁都能读——生成阶段、两个判分脚本都会更新它；
`watch.py` 只读它并刷新渲染，所以可以在另一个终端里看实时进度。
"""

from __future__ import annotations

import json
import os
import time
import unicodedata

PROGRESS_FILE = "progress.json"


def width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad(text: str, w: int, align: str = "left") -> str:
    text = str(text)
    fill = max(0, w - width(text))
    return (" " * fill + text) if align == "right" else (text + " " * fill)


def bar(done: int, total: int, slots: int = 22) -> str:
    if total <= 0:
        return "░" * slots
    filled = int(round(slots * min(done, total) / total))
    return "█" * filled + "░" * (slots - filled)


def fmt_secs(secs: float | None) -> str:
    if not secs or secs < 0:
        return "—"
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s"
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"


def read_progress(run_dir: str) -> dict:
    path = os.path.join(run_dir, PROGRESS_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def write_progress(run_dir: str, **fields) -> dict:
    """合并式更新（先读后写，最后原子替换），避免多阶段互相覆盖字段。"""
    state = read_progress(run_dir)
    # 已经跑完的 run 不要被后续补跑降级回中间状态
    if state.get("stage") == "done" and "stage" in fields and fields["stage"] != "done":
        fields.pop("stage")
    state.update(fields)
    state["updated_at"] = time.time()
    path = os.path.join(run_dir, PROGRESS_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)
    os.replace(tmp, path)
    return state


def render_dashboard(run_dir: str, run_id: str = "", cols: int = 84) -> str:
    st = read_progress(run_dir)
    if not st:
        return f"（还没有进度信息，确认 run 目录是否正确：{run_dir}）"

    stage_names = {
        "init": "初始化",
        "generate": "① 批量生成",
        "judge_open": "② 开放题 AI 复核",
        "judge_rubric": "② rubric 判分",
        "score": "③ 汇总判分",
        "done": "✅ 完成",
        "failed": "❌ 失败",
    }
    stage = st.get("stage", "init")
    elapsed = time.time() - st.get("started_at", time.time())
    gen = st.get("generate") or {}
    eta = gen.get("eta")

    lines = ["═" * cols]
    title = run_id or st.get("run_id") or os.path.basename(run_dir)
    lines.append(f" {title} ｜ 阶段：{stage_names.get(stage, stage)}")
    lines.append(f" 已用 {fmt_secs(elapsed)} ｜ 预计剩余 {fmt_secs(eta) if eta else '—'} ｜ "
                 f"更新于 {time.strftime('%H:%M:%S', time.localtime(st.get('updated_at', time.time())))}")
    lines.append("═" * cols)

    if gen:
        done, total = gen.get("done", 0), gen.get("total", 0)
        pct = f"{100 * done / total:.0f}%" if total else "—"
        lines.append("")
        lines.append(f" 生成  [{bar(done, total)}] {done}/{total}  {pct}  "
                     f"{gen.get('tps', 0):.0f} tok/s"
                     + (f"  第 {gen.get('batch')}/{gen.get('batches')} 批" if gen.get("batches") else ""))
        if gen.get("batch_budget"):
            produced = gen.get("batch_tokens", 0)
            budget = gen["batch_budget"]
            lines.append(f"   本批进行中：{gen.get('batch_type', '')} ×{gen.get('batch_size', 0)} ｜ "
                         f"已产出 {produced} tokens（上限 {budget}，{100 * produced / max(budget, 1):.0f}%）｜ "
                         f"已用 {gen.get('batch_elapsed', 0):.0f}s")

    jo, jr = st.get("judge_open") or {}, st.get("judge_rubric") or {}
    if jo or jr:
        lines.append("")
        if jo and jo.get("total"):
            d, t = jo.get("done", 0), jo.get("total", 0)
            lines.append(f" AI 复核（开放题） [{bar(d, t, 18)}] {d}/{t}  "
                         f"对 {jo.get('correct', 0)} ｜ 部分对 {jo.get('partial', 0)} ｜ "
                         f"错 {jo.get('wrong', 0)}" + (f" ｜ 规则快路 {jo.get('rule', 0)}" if jo.get("rule") else ""))
        if jr and jr.get("total"):
            d, t = jr.get("done", 0), jr.get("total", 0)
            lines.append(f" rubric 判分      [{bar(d, t, 18)}] {d}/{t}  "
                         f"平均归一化分 {jr.get('mean', 0):.1%} ｜ 踩雷题 {jr.get('negative_items', 0)}")

    dims = st.get("by_dimension") or {}
    if dims:
        lines.append("")
        lines.append(" 维度      生成进度              已判   当前准确率")
        for dim, agg in dims.items():
            d, t = agg.get("done", 0), agg.get("total", 0)
            scored = agg.get("scored", 0)
            acc = f"{agg['correct'] / scored:.1%}" if scored else "—"
            if t:
                lines.append(f" {pad(dim, 6)} {bar(d, t, 14)} {d:>3}/{t:<3}      "
                             f"{scored:>4}   {acc:>7}")

    recent = st.get("recent") or []
    if recent:
        lines.append("")
        lines.append(" 最近完成：")
        for item in recent[-6:]:
            score = item.get("score")
            score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
            lines.append(f"   {pad(item.get('id', ''), 9)} {pad(item.get('task', ''), 18)} "
                         f"{pad(item.get('verdict', ''), 8)} {score_txt:>5}  "
                         f"{pad((item.get('note') or '')[:34], 34)}")

    head = st.get("headline") or {}
    overall = st.get("summary") or {}
    if head:
        def pct(v):
            return "—" if v is None else f"{v * 100:.1f}%"
        lines.append("")
        lines.append(" 最终结果：" + " ｜ ".join([
            f"严格准确率 {pct(head.get('accuracy'))}",
            f"平均得分 {pct(overall.get('score_mean'))}",
            f"安全合规 {pct(head.get('safety_compliance'))}",
            f"引用可追溯 {pct(head.get('citation_traceability'))}（知识库指标）",
        ]))

    lines.append("═" * cols)
    return "\n".join(lines)


def render_compact(run_dir: str, run_id: str = "") -> str:
    """一行式摘要，多 run 看板里用来列出每个 run 的状态。"""
    st = read_progress(run_dir)
    name = run_id or st.get("run_id") or os.path.basename(run_dir.rstrip("/"))
    if not st:
        return f" {pad(name, 16)} 尚未开始"
    stage_names = {
        "init": "初始化", "generate": "① 生成", "judge_open": "② 开放题判分",
        "judge_rubric": "② rubric 判分", "score": "③ 汇总", "done": "✅ 完成",
        "failed": "❌ 失败",
    }
    stage = st.get("stage", "init")
    gen = st.get("generate") or {}
    done, total = gen.get("done", 0), gen.get("total", 0)
    dims = st.get("by_dimension") or {}
    scored = sum(d.get("scored", 0) for d in dims.values())
    correct = sum(d.get("correct", 0) for d in dims.values())
    acc = f"{correct / scored:.1%}" if scored else "—"
    jo, jr = st.get("judge_open") or {}, st.get("judge_rubric") or {}
    judge_txt = ""
    if jo.get("total"):
        judge_txt += f" 开放题 {jo.get('done', 0)}/{jo.get('total')}"
    if jr.get("total"):
        judge_txt += f" rubric {jr.get('done', 0)}/{jr.get('total')}"
    if stage == "done":
        head = st.get("headline") or {}
        overall = st.get("summary") or {}
        acc_v = head.get("accuracy")
        mean_v = overall.get("score_mean")
        acc_txt = "—" if acc_v is None else "%.1f%%" % (acc_v * 100)
        mean_txt = "—" if mean_v is None else "%.1f%%" % (mean_v * 100)
        return f" {pad(name, 16)} ✅ 完成 ｜ 严格准确率 {acc_txt} ｜ 平均得分 {mean_txt}"
    if total:
        return (f" {pad(name, 16)} {pad(stage_names.get(stage, stage), 12)} "
                f"{bar(done, total, 14)} {done:>3}/{total:<3} 实时准确率 {acc}{judge_txt}")
    return f" {pad(name, 16)} {stage_names.get(stage, stage)}{judge_txt}"
