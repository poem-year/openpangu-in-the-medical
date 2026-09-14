# -*- coding: utf-8 -*-
"""用 B 线智能体跑正式测评集的生成阶段（与基线同题、同判分，只是换成走智能体）。

与 run_eval.py 的 L0–L5 区别：
- L0–L5 由评测端自己拼 prompt 直接喂模型（`PanguModel.chat_batch`）；
- 本脚本把每道题**当成用户的一轮提问**交给 `agent.answer()`：由智能体自己决定要不要
  调工具（含知识库检索）、走前置安全扫描与审查层，产出 `AgentResult`。评测端只做三件事：
  给题目、收回答、落盘成与 run_eval 相同格式的 predictions.jsonl，后续 judge / score /
  show_report 原样复用。

预算不在这里控制，而在模型服务端（`scripts/pangu_server.py` 的 `--tool-max-tokens` /
`--finalize-max-tokens`）；本脚本不对生成过程加任何额外限制。

用法（模型服务需先起：bash scripts/serve_pangu.sh start）：
    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 MODEL_NAME=openpangu-7b \
      .venv/bin/python eval/run_agent_eval.py --run-id AGENT-slow-4x
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from agent.agent import answer as agent_answer  # noqa: E402
from agent.config import get_config  # noqa: E402
from agent.errors import AgentUnavailableError  # noqa: E402
from eval_lib import DATASET_DIR, PROMPT_VERSION, build_prompt  # noqa: E402
from run_eval import GENERATION_PROTOCOL as EVAL_GENERATION_PROTOCOL  # noqa: E402
from run_eval import config_fingerprint, pick_items  # noqa: E402

GENERATION_PROTOCOL = "agent-v1"


def _progress(run_dir: str, run_id: str, done: int, total: int, started: float,
              recent: list[dict]) -> None:
    payload = {
        "run_id": run_id,
        "stage": "generate" if done < total else "generate-done",
        "started_at": started,
        "generate": {"done": done, "total": total, "batch": done, "batches": total,
                     "tps": 0.0, "eta": None},
        "by_dimension": {},
        "recent": recent[-8:],
        "judge_open": {"done": 0, "total": 0},
        "judge_rubric": {"done": 0, "total": 0},
        "updated_at": time.time(),
    }
    with open(os.path.join(run_dir, "progress.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


class _PickArgs:
    """给 run_eval.pick_items 用的参数壳：只取题，不做任何生成相关设置。"""

    layer = "AGENT"
    dimensions = None
    levels = None
    task_types = None
    dataset_dir = DATASET_DIR

    def __init__(self, dataset_dir: str, per_dimension: int | None, limit: int | None) -> None:
        self.dataset_dir = dataset_dir
        self.per_dimension = per_dimension
        self.limit = limit


def main() -> int:
    parser = argparse.ArgumentParser(description="用智能体跑测评集（生成阶段）")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "eval" / "runs"))
    parser.add_argument("--run-id", default="AGENT-slow")
    parser.add_argument("--per-dimension", type=int, default=None, help="每维度只取前 N 题（冒烟）")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    todo, skipped = pick_items(_PickArgs(args.dataset_dir, args.per_dimension, args.limit))
    run_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "skipped.jsonl"), "w", encoding="utf-8") as fh:
        for item in skipped:
            fh.write(json.dumps({"id": item["id"], "dimension_code": item["dimension_code"],
                                 "task_type": item["task_type"],
                                 "skip_reason": item["skip_reason"]},
                                ensure_ascii=False) + "\n")

    cfg = get_config()
    # 指纹必须与后续 `run_eval.py --stage score --layer AGENT --thinking slow` 算出来的一致，
    # 否则判分阶段会把所有预测都当成"别的配置"过滤掉（曾经静默算出 0 题）。
    fingerprint = config_fingerprint(
        layer="AGENT",
        thinking="slow",
        batch_size=32,
        prompt_version=PROMPT_VERSION,
        generation_protocol=EVAL_GENERATION_PROTOCOL,
        budget_scale=1.0,
        budget_cap=0,
        think_ratio=1.0,
        dataset=os.path.basename(args.dataset_dir),
    )
    print(f"[agent-eval] 待跑 {len(todo)} 题，跳过 {len(skipped)} 题")
    print(f"[agent-eval] 模型服务 {cfg.openai_base_url}（模型 {cfg.model_name}）")
    print(f"[agent-eval] 客户端超时 {cfg.request_timeout}s／轮数上限 {cfg.max_turns}"
          f"／模型调用上限 {cfg.model_call_limit}／工具调用上限 {cfg.tool_call_limit}")

    pred_path = os.path.join(run_dir, "predictions.jsonl")
    done_ids: set[str] = set()
    if os.path.exists(pred_path):
        for line in open(pred_path, encoding="utf-8"):
            line = line.strip()
            if line:
                done_ids.add(json.loads(line)["id"])
        print(f"[agent-eval] 已有 {len(done_ids)} 题完成，断点续跑")

    started = time.time()
    done = len(done_ids)
    recent: list[dict] = []
    with open(pred_path, "a", encoding="utf-8") as fh:
        for item in todo:
            if item["id"] in done_ids:
                continue
            # 与基线同题：题干按 L0 口径拼（不注入检索结果，检索由智能体自己决定）
            question = build_prompt(item, layer="L0")
            t0 = time.time()
            try:
                result = agent_answer(question, [], config=cfg)
                kb_refs = [ev.ref for ev in result.assessment.evidence if ev.source == "kb"]
                record = {
                    "content": result.reply,
                    "truncated": False,
                    "risk_level": result.risk_level,
                    "handled_as": result.handled_as,
                    "used_tools": result.used_tools,
                    "confidence": result.assessment.confidence,
                    "hypotheses": [h.model_dump() for h in result.assessment.hypotheses],
                    "retrieved_chunk_ids": kb_refs,
                }
            except AgentUnavailableError as exc:
                print(f"[agent-eval] {item['id']} 智能体不可用：{exc}", flush=True)
                record = {"content": "", "truncated": True, "error": str(exc),
                          "retrieved_chunk_ids": []}

            fh.write(json.dumps({
                "id": item["id"],
                "dimension_code": item["dimension_code"],
                "task_type": item["task_type"],
                "layer": "AGENT",
                "fingerprint": fingerprint,
                "prompt_version": PROMPT_VERSION,
                "generation_protocol": GENERATION_PROTOCOL,
                "thinking_mode": "agent",
                "elapsed_seconds": round(time.time() - t0, 1),
                **record,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            done += 1
            recent.append({"id": item["id"], "task": item["task_type"],
                           "verdict": record.get("handled_as", ""),
                           "note": str(record.get("content", "")).replace("\n", " ")[:40]})
            if done % 5 == 0:
                _progress(run_dir, args.run_id, done, len(todo), started, recent)
                rate = (time.time() - started) / max(done - len(done_ids), 1)
                print(f"[agent-eval] {done}/{len(todo)} ｜ 平均 {rate:.1f}s/题 ｜ "
                      f"预计剩余 {rate * (len(todo) - done) / 60:.0f} 分钟", flush=True)
    _progress(run_dir, args.run_id, done, len(todo), started, recent)
    print(f"[agent-eval] 完成：{done} 题 → {pred_path}")
    print("[agent-eval] 接着判分与出报告：\n"
          f"  .venv-pangu/bin/python eval/run_eval.py --stage judge --run-id {args.run_id} "
          f"--layer AGENT --thinking slow --dataset-dir {args.dataset_dir}\n"
          f"  .venv-pangu/bin/python eval/run_eval.py --stage score --run-id {args.run_id} "
          f"--layer AGENT --thinking slow --dataset-dir {args.dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
