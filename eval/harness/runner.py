"""统一入口：python -m harness.runner --config eval/configs/agent-4x.yml

它只做四件事：**建不可变 run 目录 → 依次调引擎 → 归档报告 → 追加登记表**。
具体怎么生成、怎么判分仍由 `eval/` 下的引擎负责，框架不重写算法。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # 允许 `python eval/harness/runner.py` 直接跑
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "harness"

from . import store  # noqa: E402
from .config import PROJECT_ROOT, ExperimentConfig, available_configs, load_config  # noqa: E402

PY_AGENT = str(PROJECT_ROOT / ".venv" / "bin" / "python")
PY_MODEL = str(PROJECT_ROOT / ".venv-pangu" / "bin" / "python")


def _stages(cfg: ExperimentConfig, run_id: str, run_dir: Path) -> list[list[str]]:
    """把实验配置翻译成一组命令（顺序执行）。"""
    dataset = str(PROJECT_ROOT / "正式测评集")
    agent_flags = ["--run-id", run_id, "--batch-size", str(cfg.batch_size)]
    if cfg.per_dimension:
        agent_flags += ["--per-dimension", str(cfg.per_dimension)]
    if cfg.limit:
        agent_flags += ["--limit", str(cfg.limit)]

    direct_flags = ["--run-id", run_id, "--layer", cfg.layer, "--thinking", cfg.thinking,
                    "--batch-size", str(cfg.batch_size), "--dataset-dir", dataset,
                    "--budget-scale", str(cfg.budget_scale)]
    if cfg.per_dimension:
        direct_flags += ["--per-dimension", str(cfg.per_dimension)]
    if cfg.limit:
        direct_flags += ["--limit", str(cfg.limit)]
    if cfg.dimensions:
        direct_flags += ["--dimensions", *cfg.dimensions]

    if cfg.engine == "agent-batch":
        commands = [[PY_AGENT, "eval/run_agent_batch.py", *agent_flags]]
        score_flags = ["--run-id", run_id, "--layer", "AGENT", "--thinking", "slow",
                       "--dataset-dir", dataset]
    else:
        commands = [[PY_MODEL, "eval/run_eval.py", "--stage", "generate", *direct_flags]]
        score_flags = direct_flags[:]

    if cfg.judge:
        commands.append([PY_MODEL, "eval/run_eval.py", "--stage", "judge", *score_flags,
                         "--judge-workers", str(cfg.judge_workers)])
    commands.append([PY_MODEL, "eval/run_eval.py", "--stage", "score", *score_flags])
    return commands


def _env_for(cfg: ExperimentConfig) -> dict:
    env = dict(os.environ)
    env.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    env.setdefault("MODEL_NAME", "openpangu-7b")
    env.setdefault("REQUEST_TIMEOUT", "900")
    # 预算按倍数换算到服务端的两个生成口；agent-batch 由环境变量读
    env["TOOL_MAX_TOKENS"] = str(int(320 * cfg.budget_scale))
    env["FINALIZE_MAX_TOKENS"] = str(int(900 * cfg.budget_scale))
    env["AGENT_RETRIEVAL_K"] = str(cfg.retrieval_k)
    env["AGENT_BATCH_SIZE"] = str(cfg.batch_size)
    return env


def _summarize(run_dir: Path) -> dict:
    """从一次运行的产物里取登记表要用的数（读不到就给空，不编）。"""
    row: dict = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        headline = summary.get("headline") or {}
        row["accuracy"] = _pct(headline.get("accuracy"))
        row["safety_compliance"] = _pct(headline.get("safety_compliance"))
        row["citation_traceability"] = _pct(headline.get("citation_traceability"))
    scores_path = run_dir / "scores.jsonl"
    if scores_path.exists():
        values = []
        ids = set()
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            ids.add(record["id"])
            if record.get("score") is not None:
                values.append(float(record["score"]))
        row["n_items"] = len(ids)
        row["mean_score"] = _pct(sum(values) / len(values)) if values else ""
    return row


def _pct(value) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="长期测评框架统一入口")
    parser.add_argument("--config", default=None, help="实验配置（eval/configs/*.yml）")
    parser.add_argument("--limit", type=int, default=None, help="临时改总题数上限")
    parser.add_argument("--per-dimension", type=int, default=None, help="临时改每维度题量")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--budget-scale", type=float, default=None)
    parser.add_argument("--retrieval-k", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true", help="跳过 AI 判分（只出规则判分）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    parser.add_argument("--list", action="store_true", help="列出可用配置")
    args = parser.parse_args()

    if args.list:
        for path in available_configs():
            cfg = load_config(path)
            print(f"{path.name:24s} {cfg.name:22s} {cfg.engine:12s} {cfg.param_summary()}")
        return 0
    if not args.config:
        parser.error("需要 --config（或用 --list 看可用配置）")

    cfg = load_config(args.config)
    for field in ("limit", "per_dimension", "batch_size", "budget_scale", "retrieval_k"):
        override = getattr(args, field)
        if override is not None:
            setattr(cfg, field, override)
    if args.no_judge:
        cfg.judge = False
    cfg = cfg.validated()

    run_id = store.new_run_id(cfg.name, cfg.fingerprint_payload())
    run_dir = store.RUNS_DIR / run_id
    commands = _stages(cfg, run_id, run_dir)

    print(f"实验：{cfg.name}（{cfg.engine}）｜参数：{cfg.param_summary()}")
    print(f"run-id：{run_id}")
    print(f"过程数据：{run_dir}")
    print(f"报告归档：{store.REPORTS_DIR / run_id}")
    for command in commands:
        print("  $ " + " ".join(command))
    if args.dry_run:
        return 0

    store.create_run(run_id, {
        "config_file": str(Path(args.config).resolve()),
        "config": cfg.__dict__,
        "params": cfg.param_summary(),
        "fingerprint": store.short_fingerprint(cfg.fingerprint_payload()),
        "commands": [" ".join(c) for c in commands],
    })
    log_path = run_dir / "harness.log"
    env = _env_for(cfg)
    with log_path.open("a", encoding="utf-8") as log:
        for command in commands:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            proc = subprocess.run(command, cwd=PROJECT_ROOT, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, check=False)
            if proc.returncode != 0:
                store.finish_run(run_dir, status="failed", note=f"退出码 {proc.returncode}")
                print(f"✗ 阶段失败（退出码 {proc.returncode}）：{' '.join(command)}\n  日志：{log_path}")
                return proc.returncode

    report_dir = store.archive_report(run_id)
    row = {
        "run_id": run_id,
        "finished_at": store.read_meta(run_dir).get("finished_at_iso", ""),
        "name": cfg.name, "engine": cfg.engine, "params": cfg.param_summary(),
        "report": f"[报告]({run_id}/summary.md)",
        **_summarize(run_dir),
    }
    store.append_index(row)
    store.finish_run(run_dir, status="done", note="报告已归档")
    print(f"✓ 完成：{run_id}")
    print(f"  过程数据 {run_dir}")
    print(f"  报告     {report_dir}")
    print(f"  登记表   {store.INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
