"""结果落盘：不可变的 run 目录 + 只追加的报告登记表。

规矩只有一条：**任何一次运行都写进新目录，绝不覆盖历史**。
run id 形如 `20260914T152007_agent-batch-4x_a1b2c3d4`：
UTC 时间戳保证唯一，实验名保证可读，8 位指纹保证「同名不同参数」也能区分。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "eval" / "runs"
REPORTS_DIR = PROJECT_ROOT / "评测报告"
INDEX_PATH = REPORTS_DIR / "INDEX.md"


class RunExistsError(RuntimeError):
    """目标 run 目录已存在——按约定不覆盖，直接报错。"""


def short_fingerprint(payload: dict) -> str:
    """配置指纹：同配置得到同值，用于 run id 与「这轮是不是同一个实验」的判断。"""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]


def new_run_id(name: str, payload: dict, *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}_{name}_{short_fingerprint(payload)}"


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - 没有 git 也能跑
        return ""


def create_run(run_id: str, meta: dict, *, runs_dir: Path | str = RUNS_DIR) -> Path:
    """建 run 目录并写 meta.json；目录已存在则报错（不覆盖）。"""
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists():
        raise RunExistsError(
            f"run 目录已存在，拒绝覆盖：{run_dir}\n"
            "换一个 run-id，或删掉/改名旧目录（框架本身不会覆盖任何历史结果）。"
        )
    run_dir.mkdir(parents=True)
    meta = {
        "run_id": run_id,
        "started_at": time.time(),
        "started_at_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "status": "running",
        **meta,
    }
    write_meta(run_dir, meta)
    return run_dir


def write_meta(run_dir: Path | str, meta: dict) -> None:
    path = Path(run_dir) / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def read_meta(run_dir: Path | str) -> dict:
    path = Path(run_dir) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def finish_run(run_dir: Path | str, *, status: str, note: str = "") -> dict:
    meta = read_meta(run_dir)
    meta.update({"status": status, "finished_at": time.time(),
                 "finished_at_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "note": note})
    if meta.get("started_at"):
        meta["elapsed_seconds"] = round(time.time() - float(meta["started_at"]), 1)
    write_meta(run_dir, meta)
    return meta


def archive_report(run_id: str, *, runs_dir: Path | str = RUNS_DIR,
                   reports_dir: Path | str = REPORTS_DIR) -> Path | None:
    """把一次运行的报告类文件快照到 `评测报告/<run-id>/`（不覆盖已有文件）。"""
    src = Path(runs_dir) / run_id
    dst = Path(reports_dir) / run_id
    if not src.is_dir():
        return None
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("summary.md", "summary.json", "report.html", "compare.md",
                 "skipped.jsonl", "meta.json"):
        source = src / name
        target = dst / name
        if source.is_file() and not target.exists():
            target.write_bytes(source.read_bytes())
    return dst


def append_index(row: dict, *, index_path: Path | str = INDEX_PATH) -> None:
    """把一行登记写进 `INDEX.md`（只追加，不改历史行）。"""
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# 评测登记表\n\n"
            "> 由 `eval/harness` 自动追加，**只增不改**。每行一次运行，历史结果不覆盖。\n\n"
            "| run-id | 完成时间(UTC) | 实验 | 层级/引擎 | 关键参数 | 题目 | 准确率 | 均分 | 安全合规 | 引用可追溯 | 报告 |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
            encoding="utf-8",
        )
    cells = " | ".join(str(row.get(key, "")) for key in (
        "run_id", "finished_at", "name", "engine", "params", "n_items",
        "accuracy", "mean_score", "safety_compliance", "citation_traceability", "report",
    ))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"| {cells} |\n")
