# -*- coding: utf-8 -*-
"""正式测评集评测入口：生成 → 判分 → 报告（可断点续跑）。

用法（只看估时，不实际跑）：
    source /data/openpangu/scripts/pangu_env.sh
    /data/openpangu/.venv-pangu/bin/python eval/run_eval.py --stage estimate

跑全量（332 题里可自动跑的 312 题）：
    /data/openpangu/.venv-pangu/bin/python eval/run_eval.py --stage all --batch-size 32

只跑 L0 核心题（110 题：D01/D02 选择 + D05 开放诊断）：
    /data/openpangu/.venv-pangu/bin/python eval/run_eval.py --stage all --levels L0 --batch-size 32

小样本验证流程（每维度 2 题）：
    /data/openpangu/.venv-pangu/bin/python eval/run_eval.py --stage all --per-dimension 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from eval_lib import (  # noqa: E402
    DATASET_DIR,
    MAX_NEW_TOKENS,
    PROMPT_VERSION,
    SKIP_REASONS,
    build_prompt,
    config_fingerprint,
    layer_has_rag,
    load_items,
    score_item,
    wilson_ci,
)
from pangu_infer import PanguModel  # noqa: E402
from progress import bar as progress_bar, write_progress  # noqa: E402
from show_report import write_html  # noqa: E402

# 生成阶段就能即时判分的题型（用来在跑的过程中显示实时准确率）
INSTANT_TASKS = {"mcq_single", "calculation", "evidence_qa"}


def pick_items(args) -> tuple[list[dict], list[dict]]:
    """选待跑题目；返回 (要跑的, 跳过的)。"""
    items = load_items(
        dataset_dir=args.dataset_dir,
        dimensions=args.dimensions,
        levels=args.levels,
        task_types=args.task_types,
    )
    if args.per_dimension:
        by_dim: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            by_dim[it["dimension_code"]].append(it)
        items = [it for dim in sorted(by_dim) for it in by_dim[dim][: args.per_dimension]]
    if args.limit:
        items = items[: args.limit]

    todo, skipped = [], []
    for it in items:
        if it["task_type"] in SKIP_REASONS:
            skipped.append({**it, "skip_reason": SKIP_REASONS[it["task_type"]]})
        else:
            todo.append(it)
    return todo, skipped


def estimate(todo: list[dict], thinking: str, batch_size: int) -> dict:
    """按题型预算估输出 token 与耗时（吞吐取实测经验值）。"""
    scale = 1.0 if thinking == "slow" else 0.75
    planned = 0
    for it in todo:
        planned += int(MAX_NEW_TOKENS.get(it["task_type"], 512) * scale)
    # 实测：慢思考 batch32 ≈ 700 tok/s，快思考更高；按 60% 有效输出（提前 EOS）估
    throughput = 700.0 if thinking == "slow" else 1000.0
    est_tokens = planned * 0.6
    est_seconds = est_tokens / throughput
    return {
        "items": len(todo),
        "planned_max_tokens": planned,
        "est_output_tokens": int(est_tokens),
        "est_seconds": round(est_seconds, 1),
        "throughput_assumed": throughput,
        "batch_size": batch_size,
    }


def token_budget(task_type: str, thinking: str) -> int:
    """输出预算：慢思考用全量（会先写推理），快思考按 60% 给但保底 640。"""
    base = MAX_NEW_TOKENS.get(task_type, 512)
    if thinking == "slow":
        return base
    return max(640, int(base * 0.6))


def budget_for(task_type: str, args) -> int:
    """在基准预算上套用 --budget-scale / --budget-cap（用于重跑截断题）。"""
    base = token_budget(task_type, args.thinking)
    scale = float(getattr(args, "budget_scale", 1.0) or 1.0)
    cap = int(getattr(args, "budget_cap", 0) or 0)
    out = int(base * scale)
    if cap:
        out = min(out, cap)
    return max(base, out)


def scan_done(pred_path: str, fingerprint: str) -> tuple[set[str], set[str]]:
    """扫已完成的预测：返回 (有效的题号, 上次被截断需要重跑的题号)。

    有效 = 配置指纹一致且没有触顶截断；被截断的题要按调大后的预算重跑。
    """
    done: set[str] = set()
    truncated: set[str] = set()
    if not os.path.exists(pred_path):
        return done, truncated
    with open(pred_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("fingerprint") != fingerprint:
                continue
            iid = rec.get("id")
            if not iid:
                continue
            if rec.get("truncated"):
                done.discard(iid)
                truncated.add(iid)
            else:
                done.add(iid)
                truncated.discard(iid)
    return done, truncated


def stage_generate(items: list[dict], args, run_dir: str, fingerprint: str) -> None:
    pred_path = os.path.join(run_dir, "predictions.jsonl")
    done, truncated_prev = scan_done(pred_path, fingerprint)
    pending = [it for it in items if it["id"] not in done]
    if truncated_prev:
        print(f"[generate] 有 {len(truncated_prev)} 题上次被截断，本次用调大后的预算重跑")
    print(f"[generate] 已完成 {len(done)} 题，待跑 {len(pending)} 题，配置指纹 {fingerprint}")
    if not pending:
        return

    # 排序：按题型分组，组内按长度排序。这样每一批都是同题型、同预算，
    # 既减少 padding，也避免「长题型被短题型的预算截断」。
    ordered = sorted(
        pending,
        key=lambda it: (
            it["task_type"],
            budget_for(it["task_type"], args),
            len(build_prompt(it, args.layer)),
            it["id"],
        ),
    )
    for it in ordered:
        it["_prompt"] = build_prompt(it, args.layer)

    model = PanguModel()
    model.load()
    # 预热：昇腾算子首次调用要编译，放在计时之外，避免第一批被拖慢
    t_warm = time.time()
    model.chat_batch(["预热"], system_prompt=None, fast_thinking=True, max_new_tokens=8)
    print(f"[generate] 预热完成 {time.time() - t_warm:.1f}s", flush=True)

    # 按题型切批：一批之内只有一个题型，保证输出预算按该题型给足
    batches: list[list[dict]] = []
    for task in sorted({it["task_type"] for it in ordered}):
        of_task = [it for it in ordered if it["task_type"] == task]
        for start in range(0, len(of_task), args.batch_size):
            batches.append(of_task[start:start + args.batch_size])

    dim_state: dict[str, dict] = {}
    for it in items:
        dim_state.setdefault(it["dimension_code"], {"done": 0, "total": 0, "scored": 0, "correct": 0})
    for it in items:
        dim_state[it["dimension_code"]]["total"] += 1
    write_progress(
        run_dir,
        run_id=os.path.basename(run_dir),
        stage="generate",
        started_at=time.time(),
        generate={"done": len(done), "total": len(items), "batch": 0,
                  "batches": len(batches), "tps": 0.0, "eta": None},
        by_dimension=dim_state,
        recent=[],
        judge_open={"done": 0, "total": 0},
        judge_rubric={"done": 0, "total": 0},
    )

    written = 0
    t_start = time.time()
    recent: list[dict] = []
    with open(pred_path, "a", encoding="utf-8") as fh:
        for batch_idx, batch in enumerate(batches, start=1):
            group = batch[0]["task_type"]
            max_new = budget_for(group, args)
            t0 = time.time()
            batch_budget = len(batch) * max_new
            write_progress(
                run_dir,
                stage="generate",
                generate={"done": len(done) + written, "total": len(done) + len(pending),
                          "batch": batch_idx, "batches": len(batches), "tps": 0.0,
                          "eta": None, "batch_type": group, "batch_size": len(batch),
                          "batch_tokens": 0, "batch_budget": batch_budget,
                          "batch_elapsed": 0.0},
            )
            last_write = [0.0]

            def on_step(generated_tokens: int, _t0=t0, _budget=batch_budget, _idx=batch_idx) -> None:
                """批内进度：每 2 秒最多写一次状态，避免频繁 IO 拖慢生成。"""
                now = time.time()
                if now - last_write[0] < 2.0:
                    return
                last_write[0] = now
                write_progress(
                    run_dir,
                    generate={"done": len(done) + written,
                              "total": len(done) + len(pending),
                              "batch": _idx, "batches": len(batches), "tps": 0.0,
                              "eta": None, "batch_type": group, "batch_size": len(batch),
                              "batch_tokens": generated_tokens, "batch_budget": _budget,
                              "batch_elapsed": round(now - _t0, 1)},
                )

            out = model.chat_batch(
                [it["_prompt"] for it in batch],
                system_prompt=None,
                fast_thinking=(args.thinking == "fast"),
                max_new_tokens=max_new,
                progress_cb=on_step,
            )
            elapsed = time.time() - t0
            for it, res in zip(batch, out["results"]):
                fh.write(json.dumps({
                    "id": it["id"],
                    "dimension_code": it["dimension_code"],
                    "task_type": it["task_type"],
                    "layer": args.layer,
                    "fingerprint": fingerprint,
                    "prompt_version": PROMPT_VERSION,
                    "thinking_mode": args.thinking,
                    "max_new_tokens": max_new,
                    "budget_scale": float(getattr(args, "budget_scale", 1.0) or 1.0),
                    "content": res["content"],
                    "thinking": res["thinking"],
                    "raw": res["raw"],
                    "output_tokens": res["output_tokens"],
                    "truncated": res["output_tokens"] >= max_new,
                }, ensure_ascii=False) + "\n")
                written += 1
                dim = dim_state[it["dimension_code"]]
                dim["done"] += 1
                note, verdict, score_value = "", "", None
                if it["task_type"] in INSTANT_TASKS:
                    sc = score_item(it, res["content"], res.get("thinking", ""))
                    dim["scored"] += 1
                    if sc.correct:
                        dim["correct"] += 1
                    score_value = sc.score
                    verdict = "对" if sc.correct else "错"
                    note = sc.detail
                else:
                    note = (res["content"] or "").replace("\n", " ")[:40]
                recent.append({"id": it["id"], "task": it["task_type"], "verdict": verdict,
                               "score": score_value, "note": note})
            fh.flush()
            done_n = len(done) + written
            total_n = len(done) + len(pending)
            rate = out["total_output_tokens"] / max(elapsed, 1e-6)
            eta = (total_n - done_n) * (elapsed / max(len(batch), 1))
            write_progress(
                run_dir,
                stage="generate",
                generate={"done": done_n, "total": total_n, "batch": batch_idx,
                          "batches": len(batches), "tps": round(rate, 1), "eta": round(eta, 1)},
                by_dimension=dim_state,
                recent=recent[-8:],
            )
            scored_all = sum(d["scored"] for d in dim_state.values())
            correct_all = sum(d["correct"] for d in dim_state.values())
            acc_txt = f"{correct_all / scored_all:.1%}" if scored_all else "—"
            print(
                f"[generate] [{progress_bar(done_n, total_n, 24)}] {done_n}/{total_n} "
                f"{100 * done_n / max(total_n, 1):.0f}% | {group} ×{len(batch)} | "
                f"{rate:.0f} tok/s | 已判 {scored_all} 题准确率 {acc_txt} | "
                f"已用 {time.time() - t_start:.0f}s 预计还需 {eta:.0f}s",
                flush=True,
            )


def summarize(rows: list[dict]) -> dict:
    """按维度汇总：准确率（可判分的题）+ Wilson 区间 + 待判分数量。"""
    per_dim: dict[str, dict] = {}
    for row in rows:
        dim = row["dimension_code"]
        agg = per_dim.setdefault(
            dim,
            {"n": 0, "scored": 0, "correct": 0, "pending": 0, "truncated": 0,
             "score_sum": 0.0,
             "by_task": defaultdict(lambda: {"n": 0, "correct": 0})},
        )
        agg["n"] += 1
        if row.get("truncated"):
            agg["truncated"] += 1
        task = agg["by_task"][row["task_type"]]
        task["n"] += 1
        if row["score"] is None:
            agg["pending"] += 1
            continue
        agg["scored"] += 1
        agg["score_sum"] += float(row["score"])
        if row["correct"]:
            agg["correct"] += 1
            task["correct"] += 1

    out: dict = {"dimensions": {}, "overall": {}}
    tot_scored = tot_correct = tot_n = tot_pending = tot_truncated = 0
    tot_score = 0.0
    for dim, agg in sorted(per_dim.items()):
        scored, correct = agg["scored"], agg["correct"]
        lo, hi = wilson_ci(correct, scored)
        out["dimensions"][dim] = {
            "n": agg["n"],
            "scored": scored,
            "correct": correct,
            "pending": agg["pending"],
            "truncated": agg["truncated"],
            "accuracy": round(correct / scored, 4) if scored else None,
            "score_mean": round(agg["score_sum"] / scored, 4) if scored else None,
            "ci95": [round(lo, 4), round(hi, 4)] if scored else None,
            "by_task": {k: dict(v) for k, v in agg["by_task"].items()},
        }
        tot_scored += scored
        tot_correct += correct
        tot_n += agg["n"]
        tot_pending += agg["pending"]
        tot_truncated += agg["truncated"]
        tot_score += agg["score_sum"]
    lo, hi = wilson_ci(tot_correct, tot_scored)
    out["overall"] = {
        "n": tot_n,
        "scored": tot_scored,
        "correct": tot_correct,
        "pending": tot_pending,
        "truncated": tot_truncated,
        "accuracy": round(tot_correct / tot_scored, 4) if tot_scored else None,
        "score_mean": round(tot_score / tot_scored, 4) if tot_scored else None,
        "ci95": [round(lo, 4), round(hi, 4)] if tot_scored else None,
    }
    return out


def write_report(summary: dict, path: str) -> None:
    overall = summary["overall"]
    headline = summary.get("headline") or {}
    rubric = summary.get("rubric") or {}
    lines = [
        f"# 评测结果 · {summary.get('run_id', '')}",
        "",
        f"- 层级：{summary.get('layer')}　思考模式：{summary.get('thinking_mode')}　批大小：{summary.get('batch_size')}",
        f"- 配置指纹：`{summary.get('fingerprint')}`",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 准确率（严格口径） | {headline.get('accuracy')} |",
        f"| 平均得分（含部分分，0.5=方向对） | {overall.get('score_mean')} |",
        f"| 安全合规率（D09–D11 rubric，需先跑 judge） | {headline.get('safety_compliance')} |",
        f"| 引用可追溯率（知识库模块指标） | "
        f"{0.0 if not headline.get('rag_enabled') else headline.get('citation_traceability')} "
        f"（{headline.get('citation_traceability_note')}） |",
        "",
        "## 计数",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 题目数 | {overall['n']} |",
        f"| 可自动判分 | {overall['scored']} |",
        f"| 待判分（rubric/人工） | {overall['pending']} |",
        f"| 触顶截断（输出预算不够） | {overall.get('truncated', 0)} |",
        f"| 准确率 | {overall['accuracy']} |",
        f"| 95% CI | {overall['ci95']} |",
        "",
        "## 分维度",
        "",
        "| 维度 | 题量 | 可判分 | 正确 | 准确率 | 95% CI | 待判分 | 截断 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for dim, agg in summary["dimensions"].items():
        lines.append(
            f"| {dim} | {agg['n']} | {agg['scored']} | {agg['correct']} | "
            f"{agg['accuracy']} | {agg['ci95']} | {agg['pending']} | {agg.get('truncated', 0)} |"
        )
    lines += [
        "",
        "> 说明：准确率只统计「可自动判分」的题；rubric_scored（D09–D11）需要判分阶段或人工复核，未计入。",
        "> 开放题（D03/D04/D05/D06）用要点覆盖、诊断名匹配自动判分，属机器近似，正式报告建议抽样人工复核。",
    ]
    if rubric:
        lines += [
            "",
            "## rubric 判分（D09–D11）",
            "",
            "| 维度 | 题量 | 平均归一化分 | 踩雷题数 | 踩雷率 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for dim, agg in rubric.items():
            lines.append(f"| {dim} | {agg['n']} | {agg['mean_rubric_score']} | "
                         f"{agg['negative_hit_items']} | {agg['negative_hit_rate']} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def stage_score(items: list[dict], args, run_dir: str, fingerprint: str) -> dict:
    pred_path = os.path.join(run_dir, "predictions.jsonl")
    score_path = os.path.join(run_dir, "scores.jsonl")
    # AI 复核结果（如果跑过 ai_judge）：开放题以 AI 判定为准，规则判定作为兜底
    ai_judgments: dict[str, dict] = {}
    ai_path = os.path.join(run_dir, "answer_judgments.jsonl")
    if os.path.exists(ai_path):
        with open(ai_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("ok"):
                    ai_judgments[rec["id"]] = rec
    preds: dict[str, dict] = {}
    with open(pred_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("fingerprint") == fingerprint:
                preds[rec["id"]] = rec

    by_id = {it["id"]: it for it in items}
    rows = []
    with open(score_path, "w", encoding="utf-8") as fh:
        for item_id, pred in preds.items():
            item = by_id.get(item_id)
            if item is None:
                continue
            sc = score_item(item, pred["content"], pred.get("thinking", ""))
            ai = ai_judgments.get(item_id)
            if ai is not None and ai.get("score") is not None:
                sc.score = float(ai["score"])
                sc.correct = sc.score >= 1.0
                sc.metrics = {**sc.metrics, "ai_verdict": ai.get("verdict"),
                              "ai_reason": ai.get("reason"), "judge_source": ai.get("source")}
                sc.detail = f"[AI 复核/{ai.get('source')}] {ai.get('verdict')}：{ai.get('reason')}"
            row = {
                "id": sc.item_id,
                "dimension_code": sc.dimension,
                "task_type": sc.task_type,
                "score": sc.score,
                "correct": sc.correct,
                "metrics": sc.metrics,
                "detail": sc.detail,
                "output_tokens": pred.get("output_tokens"),
                "truncated": pred.get("truncated", False),
                "judge_source": (ai or {}).get("source", "rule"),
            }
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    summary["headline"] = headline_metrics(rows, run_dir, args.layer)
    summary["rubric"] = rubric_stats(run_dir)
    summary.update(
        run_id=os.path.basename(run_dir),
        fingerprint=fingerprint,
        layer=args.layer,
        thinking_mode=args.thinking,
        batch_size=args.batch_size,
    )
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    write_report(summary, os.path.join(run_dir, "summary.md"))
    write_html(summary, os.path.join(run_dir, "report.html"))
    write_progress(run_dir, stage="done", summary=summary.get("overall"),
                   headline=summary.get("headline"))
    return summary


def headline_metrics(rows: list[dict], run_dir: str, layer: str = "L0") -> dict:
    """上报指标口径。

    - 准确率：机器/AI 判分结果（严格口径）
    - 引用可追溯率：**知识库模块专属指标**。基线（L0/L1）没有检索与引用，固定为 0；
      L2/L3/L5 接入检索后用 traceability.py 校验真实 chunk_id（可定位 + 可支撑）。
    - 安全合规率：D09–D11 rubric 里未踩负分条目的比例。

    暂不设置为上报指标的：
    - 幻觉率：口径未定，暂不设置（原始判定仍留在 scores.jsonl 里备查）。
    - D13 材料引用合规率：题目自带材料，不等于知识库溯源，不作上报指标
      （D13 的维度准确率仍然照常统计）。
    """
    auto = [r for r in rows if r["score"] is not None]
    accuracy = (sum(1 for r in auto if r["correct"]) / len(auto)) if auto else None

    # D13：材料引用合规率（内部诊断量，不对外上报）
    cite_rows = [r for r in rows if "citation_traceable" in (r.get("metrics") or {})]
    d13_citation = (
        sum(1 for r in cite_rows if r["metrics"]["citation_traceable"]) / len(cite_rows)
        if cite_rows else None
    )

    rubric = rubric_stats(run_dir)
    safe_rate = None
    if rubric:
        total = sum(v["n"] for v in rubric.values())
        dirty = sum(v["negative_hit_items"] for v in rubric.values())
        safe_rate = 1 - dirty / total if total else None

    rag = layer_has_rag(layer)
    return {
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "citation_traceability": 0.0 if not rag else None,
        "citation_traceability_note": (
            "基线无知识库、无引用，按定义固定为 0"
            if not rag else "待接入检索后用真实 chunk_id 计算（可定位 + 可支撑）"
        ),
        "safety_compliance": round(safe_rate, 4) if safe_rate is not None else None,
        "rag_enabled": rag,
        "note": "引用可追溯率是知识库模块指标，基线固定为 0；安全合规率来自 D09–D11 的 rubric 判分；"
                "幻觉率与 D13 材料引用合规率暂不列为上报指标",
    }


def rubric_stats(run_dir: str) -> dict:
    """读取 rubric 判分结果（如果已跑 judge 阶段）。"""
    path = os.path.join(run_dir, "rubric_judgments.jsonl")
    if not os.path.exists(path):
        return {}
    per_dim: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("ok"):
                continue
            dim = rec["dimension_code"]
            agg = per_dim.setdefault(dim, {"n": 0, "norm_sum": 0.0, "negative_hit_items": 0})
            agg["n"] += 1
            agg["norm_sum"] += rec.get("rubric_score_norm") or 0.0
            if rec.get("negative_hits"):
                agg["negative_hit_items"] += 1
    return {
        dim: {
            "n": v["n"],
            "mean_rubric_score": round(v["norm_sum"] / v["n"], 4),
            "negative_hit_items": v["negative_hit_items"],
            "negative_hit_rate": round(v["negative_hit_items"] / v["n"], 4),
        }
        for dim, v in sorted(per_dim.items())
    }


def stage_judge(run_dir: str, args) -> None:
    """调用 DeepSeek：D09–D11 做 rubric 判分，开放题（D03–D06）做诊断等价复核。"""
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    workers = ["--workers", str(args.judge_workers)]
    for script, extra in (("ai_judge.py", workers), ("judge.py", workers)):
        cmd = [sys.executable, os.path.join(here, script),
               "--run-dir", run_dir, "--dataset-dir", args.dataset_dir] + extra
        if args.judge_limit:
            cmd += ["--limit", str(args.judge_limit)]
        subprocess.run(cmd, check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["estimate", "generate", "judge", "score", "report", "all"])
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--out-dir", default="/data/openpangu/eval/runs")
    parser.add_argument("--layer", default="L0")
    parser.add_argument("--thinking", default="slow", choices=["slow", "fast"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dimensions", nargs="*", default=None)
    parser.add_argument("--levels", nargs="*", default=None)
    parser.add_argument("--task-types", nargs="*", default=None)
    parser.add_argument("--per-dimension", type=int, default=None, help="每维度只取前 N 题（冒烟用）")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--judge-limit", type=int, default=None, help="judge 阶段只判前 N 题（冒烟用）")
    parser.add_argument("--judge-workers", type=int, default=6, help="开放题 AI 复核并发数")
    parser.add_argument("--with-judge", action="store_true", help="all 阶段一并跑 rubric 判分（走 DeepSeek API）")
    parser.add_argument("--budget-scale", type=float, default=1.0,
                        help="重跑截断题时把输出预算放大若干倍（默认 1.0）")
    parser.add_argument("--budget-cap", type=int, default=0,
                        help="单题输出预算上限（0 表示不设上限）")
    args = parser.parse_args()

    todo, skipped = pick_items(args)
    run_id = args.run_id or f"{args.layer}-{args.thinking}-b{args.batch_size}-p{PROMPT_VERSION}"
    fingerprint = config_fingerprint(
        layer=args.layer,
        thinking=args.thinking,
        batch_size=args.batch_size,
        prompt_version=PROMPT_VERSION,
        dataset=os.path.basename(args.dataset_dir),
    )
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[plan] 待跑 {len(todo)} 题，跳过 {len(skipped)} 题（需要模拟患者/检索器）")
    if skipped:
        with open(os.path.join(run_dir, "skipped.jsonl"), "w", encoding="utf-8") as fh:
            for it in skipped:
                fh.write(json.dumps(
                    {k: it[k] for k in ("id", "dimension_code", "task_type", "skip_reason")},
                    ensure_ascii=False) + "\n")

    est = estimate(todo, args.thinking, args.batch_size)
    print(f"[plan] 预计输出约 {est['est_output_tokens']} tokens，"
          f"按 {est['throughput_assumed']} tok/s 估算约 {est['est_seconds'] / 60:.1f} 分钟"
          f"（不含模型加载与判分）")

    if args.stage == "estimate":
        return
    if args.stage in ("generate", "all"):
        stage_generate(todo, args, run_dir, fingerprint)
    if args.stage == "judge" or (args.stage == "all" and args.with_judge):
        stage_judge(run_dir, args)
    if args.stage in ("score", "report", "all"):
        summary = stage_score(todo, args, run_dir, fingerprint)
        print(f"[score] 可判分 {summary['overall']['scored']} 题，"
              f"准确率 {summary['overall']['accuracy']}，报告：{os.path.join(run_dir, 'summary.md')}")


if __name__ == "__main__":
    main()
