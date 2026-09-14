# -*- coding: utf-8 -*-
"""智能体评测的批量驱动：把「同一轮」的题打包解码，2 小时 → 十几分钟。

与 `run_agent_eval.py`（逐题调 `agent.answer()`，串行）的区别只在**执行方式**：

    run_agent_eval.py : 一题一条会话，走 /v1/chat/completions，单条解码 37 tok/s
    run_agent_batch.py: 评测端自己跑智能体的状态机，每一轮把 N 道题打包送
                        /v1/batch/chat，批解码 600+ tok/s

复用的都是智能体原有的东西：系统提示词、工具实现、TOOL/ARGS 协议、审查层、
`AgentTurnOutput` 结构；**不重写智能体逻辑**，只把「谁来调度」从服务端挪到评测端。

流程（每题）：
    题目 → 前置安全扫描（纯代码，命中直接给固定话术，不过模型）
         → 工具轮（可多轮）：模型决定调工具 / 直接作答
         → 收口轮：产出 AgentTurnOutput
         → 审查层（代码级降级/替换）
         → predictions.jsonl（与 run_eval 同格式，judge/score/show_report 直接复用）

用法（先起服务：TOOL_MAX_TOKENS=1280 FINALIZE_MAX_TOKENS=3600 bash scripts/serve_pangu.sh start）：
    .venv/bin/python eval/run_agent_batch.py --run-id AGENTB-slow-4x
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from agent.prompt_loader import load_fix_prompt, load_system_prompt, load_templates  # noqa: E402
from agent.recorder import TurnRecorder, reset_recorder, set_recorder  # noqa: E402
from agent.review import ReviewContext, run_review  # noqa: E402
from agent.schemas import AgentTurnOutput  # noqa: E402
from agent.tools import ALL_TOOLS  # noqa: E402
from agent.tools.red_flags import PrescanHit, scan_input  # noqa: E402
from eval_lib import DATASET_DIR, PROMPT_VERSION, build_prompt  # noqa: E402
from run_eval import GENERATION_PROTOCOL as EVAL_GENERATION_PROTOCOL  # noqa: E402
from run_eval import config_fingerprint, pick_items  # noqa: E402
from scripts.pangu_tool_bridge import FORCE_FINALIZE_NOTE, REQUIRE_TOOL_NOTE  # noqa: E402

GENERATION_PROTOCOL = "agent-batch-v1"
DEFAULT_BATCH_ENDPOINT = "http://127.0.0.1:8000/v1/batch/chat"
MAX_TOOL_ROUNDS = 3
TOOL_MAX_TOKENS = int(os.environ.get("TOOL_MAX_TOKENS", "1280"))
FINALIZE_MAX_TOKENS = int(os.environ.get("FINALIZE_MAX_TOKENS", "3600"))
BATCH_SIZE = int(os.environ.get("AGENT_BATCH_SIZE", "16"))
# RAG 信息量控制：检索工具最多返回几条。条数直接决定收口轮 prompt 的 prefill 长度
# （每条片段约 500 字），是「快」与「给足资料」之间的主要旋钮。
RETRIEVAL_K_CAP = int(os.environ.get("AGENT_RETRIEVAL_K", "3"))
# 预注入确定性工具：check_red_flags / timeline_calc 是纯规则函数，评测端先算好写进
# system 提示，模型就不必为它们单开一轮（省约 10 秒/批）。置 0 可关掉，恢复「全靠模型自己调」。
PREINJECT_TOOLS = os.environ.get("AGENT_PREINJECT_TOOLS", "1") != "0"

_TOOLS_BY_NAME = {tool.__name__ if hasattr(tool, "__name__") else tool.name: tool for tool in ALL_TOOLS}

_ORIGINAL_RETRIEVE = _TOOLS_BY_NAME["retrieve_evidence"]


def _capped_retrieve_evidence(query: str, k: int = RETRIEVAL_K_CAP) -> str:
    """检索工具包装：把条数封顶（模型要 5 条也只给 3 条）。"""
    try:
        wanted = min(int(k or RETRIEVAL_K_CAP), RETRIEVAL_K_CAP)
    except (TypeError, ValueError):
        wanted = RETRIEVAL_K_CAP
    return _ORIGINAL_RETRIEVE(query, max(1, wanted))


def prelude_for(question: str) -> str:
    """预注入的确定性工具结果（写进 system 提示的附加说明）。"""
    notes: list[str] = []
    try:
        red = _TOOLS_BY_NAME["check_red_flags"]([question])
    except Exception:  # noqa: BLE001
        red = ""
    if red:
        notes.append(f"【危险信号已由系统核对】{red}")
    if notes:
        notes.append("上面这一步系统已经做过，不必再调用对应工具；其它工具照常按需调用。")
    return "\n\n".join(notes)


# 检索工具换成封顶版（模型要 5 条也只给 3 条）
_TOOLS_BY_NAME["retrieve_evidence"] = _capped_retrieve_evidence


def tool_specs() -> list[dict]:
    """把智能体的工具转成 OpenAI 风格的 function 规格（服务端据此渲染工具提示）。"""
    specs: list[dict] = []
    for name, tool in _TOOLS_BY_NAME.items():
        description = (getattr(tool, "__doc__", "") or "").strip().splitlines()[0:1]
        params = {"type": "object", "properties": {}}
        signature = getattr(tool, "__annotations__", {}) or {}
        if signature:
            properties = {}
            for arg, annotation in signature.items():
                if arg == "return":
                    continue
                json_type = "integer" if annotation in (int,) else (
                    "number" if annotation in (float,) else (
                        "array" if "list" in str(annotation) else "string"))
                properties[arg] = {"type": json_type}
            params = {"type": "object", "properties": properties}
        specs.append({
            "type": "function",
            "function": {"name": name, "description": description[0] if description else name,
                         "parameters": params},
        })
    return specs


def call_batch(endpoint: str, requests: list[dict], timeout: float = 900.0) -> list[dict]:
    payload = json.dumps({"requests": requests}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    results = body.get("results")
    if not isinstance(results, list) or len(results) != len(requests):
        raise RuntimeError(f"批处理返回条数不对：{len(results or [])} vs {len(requests)}")
    return results


def prescan_result(hit: PrescanHit, templates: dict) -> dict:
    """前置扫描命中：与 agent.answer 一样，不经过模型，直接给固定话术。"""
    if hit.kind == "self_harm":
        reply, risk, handled = templates.get("self_harm", ""), "emergency", "emergency"
        steps = ["立即联系信任的人陪伴", "拨打心理援助热线 12356 或 120 求助"]
    elif hit.kind == "emergency":
        reply, risk, handled = templates.get("emergency", ""), "emergency", "emergency"
        steps = ["立即拨打 120 或前往最近的急诊科"]
    else:
        reply, risk, handled = templates.get("out_of_scope", ""), "medium", "out_of_scope"
        steps = ["前往相应专科就诊（儿科 / 妇产科 / 精神心理科）"]
    return {
        "content": reply, "truncated": False, "risk_level": risk, "handled_as": handled,
        "used_tools": [], "confidence": "low", "retrieved_chunk_ids": [],
        "prescan": hit.rule_id, "next_steps": steps,
    }


def run_tools(name: str, arguments: dict, recorder: TurnRecorder) -> str:
    """本地执行工具（毫秒级，不用往返服务端），并把调用记进 recorder。"""
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return f"错误：没有名为 {name} 的工具。"
    token = set_recorder(recorder)
    try:
        if isinstance(arguments, dict):
            return str(tool(**arguments))
        return str(tool(arguments))
    except TypeError as exc:
        return f"错误：参数不对（{exc}）。请检查后的参数重新调用。"
    except Exception as exc:  # noqa: BLE001
        return f"错误：工具执行失败（{type(exc).__name__}: {exc}）。"
    finally:
        reset_recorder(token)


def main() -> int:
    parser = argparse.ArgumentParser(description="智能体评测（批量驱动）")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "eval" / "runs"))
    parser.add_argument("--run-id", default="AGENTB-slow")
    parser.add_argument("--endpoint", default=os.environ.get("BATCH_ENDPOINT", DEFAULT_BATCH_ENDPOINT))
    parser.add_argument("--per-dimension", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    class _Pick:
        layer = "AGENT"
        dimensions = levels = task_types = None

        def __init__(self) -> None:
            self.dataset_dir = args.dataset_dir
            self.per_dimension = args.per_dimension
            self.limit = args.limit

    todo, skipped = pick_items(_Pick())
    run_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "skipped.jsonl"), "w", encoding="utf-8") as fh:
        for item in skipped:
            fh.write(json.dumps({"id": item["id"], "dimension_code": item["dimension_code"],
                                 "task_type": item["task_type"], "skip_reason": item["skip_reason"]},
                                ensure_ascii=False) + "\n")

    fingerprint = config_fingerprint(
        layer="AGENT", thinking="slow", batch_size=32, prompt_version=PROMPT_VERSION,
        generation_protocol=EVAL_GENERATION_PROTOCOL, budget_scale=1.0, budget_cap=0,
        think_ratio=1.0, dataset=os.path.basename(args.dataset_dir),
    )
    pred_path = os.path.join(run_dir, "predictions.jsonl")
    done_ids: set[str] = set()
    if os.path.exists(pred_path):
        for line in open(pred_path, encoding="utf-8"):
            line = line.strip()
            if line:
                done_ids.add(json.loads(line)["id"])
        print(f"[agentb] 已有 {len(done_ids)} 题完成，断点续跑")

    pending = [item for item in todo if item["id"] not in done_ids]
    specs = tool_specs()
    system_prompt = load_system_prompt()
    templates = load_templates()
    print(f"[agentb] 待跑 {len(pending)} 题（已跳过 {len(skipped)}）｜批大小 {args.batch_size}")
    print(f"[agentb] 端点 {args.endpoint}｜工具轮预算 {TOOL_MAX_TOKENS}｜收口轮预算 {FINALIZE_MAX_TOKENS}")

    started = time.time()
    written = 0
    round_stats: list[dict] = []
    with open(pred_path, "a", encoding="utf-8") as fh:
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            t0 = time.time()
            results: dict[str, dict] = {}
            state: dict[str, dict] = {}
            for item in chunk:
                question = build_prompt(item, layer="L0")
                hit = scan_input(question, first_turn=True)
                if hit is not None:
                    results[item["id"]] = prescan_result(hit, templates)
                else:
                    system_text = system_prompt
                    if PREINJECT_TOOLS:
                        prelude = prelude_for(question)
                        if prelude:
                            system_text = f"{system_prompt}\n\n{prelude}"
                    state[item["id"]] = {
                        "messages": [{"role": "system", "content": system_text},
                                     {"role": "user", "content": question}],
                        "draft": "", "rounds": 0, "recorder": TurnRecorder(),
                    }

            # ---- 工具轮（可多轮）：每轮把「还没到轮数上限」的题打包一起解码 ----
            while True:
                active = [sid for sid, st in state.items() if st["rounds"] < MAX_TOOL_ROUNDS]
                if not active:
                    break
                requests = []
                for sid in active:
                    st = state[sid]
                    requests.append({
                        "kind": "tool",
                        "messages": st["messages"],
                        "tools": specs,
                        "allowed_tools": list(_TOOLS_BY_NAME),
                        "max_tokens": TOOL_MAX_TOKENS,
                        # 首轮没调工具就抢答 → 与单请求路径一致的强制补偿
                        "notes": [REQUIRE_TOOL_NOTE] if st["rounds"] == 0 else [],
                    })
                round_t0 = time.time()
                st_round = active and state[active[0]]["rounds"] or 0
                out = call_batch(args.endpoint, requests)
                round_stats.append({"round": f"tool{st_round}", "n": len(active),
                                    "seconds": round(time.time() - round_t0, 1)})
                for sid, res in zip(active, out):
                    st = state[sid]
                    st["rounds"] += 1
                    call = res.get("tool_call")
                    if call:
                        tool_result = run_tools(call["name"], call.get("arguments") or {},
                                                st["recorder"])
                        st["messages"].append({
                            "role": "assistant",
                            "content": f"TOOL: {call['name']}\nARGS: "
                                       f"{json.dumps(call.get('arguments') or {}, ensure_ascii=False)}",
                        })
                        st["messages"].append({
                            "role": "tool",
                            "content": f"{call['name']} 返回：{tool_result}",
                        })
                    else:
                        # 模型选择直接回答 → 这段文本作为草稿，进收口轮
                        st["draft"] = res.get("text") or ""
                        st["rounds"] = MAX_TOOL_ROUNDS  # 不再走工具轮

            # ---- 收口轮：这一批里所有还留在 state 的题一起打包 ----
            finalize_ids = list(state)
            if finalize_ids:
                round_t0 = time.time()
                requests = []
                for sid in finalize_ids:
                    st = state[sid]
                    requests.append({
                        "kind": "finalize", "messages": st["messages"],
                        "draft_reply": st.get("draft", ""), "max_tokens": FINALIZE_MAX_TOKENS,
                        # 工具轮用满还没收口 → 提示模型这一轮必须给结论
                        "retry_note": FORCE_FINALIZE_NOTE if st["rounds"] >= MAX_TOOL_ROUNDS else "",
                    })
                out = call_batch(args.endpoint, requests)
                round_stats.append({"round": "finalize", "n": len(finalize_ids),
                                    "seconds": round(time.time() - round_t0, 1)})
                for sid, res in zip(finalize_ids, out):
                    st = state[sid]
                    payload = res.get("payload") or {}
                    try:
                        output = AgentTurnOutput.model_validate(payload)
                    except Exception:  # noqa: BLE001
                        results[sid] = {
                            "content": "抱歉，本轮我没能整理出可靠的分析。您可以换一种说法再描述一次；"
                                       "如果情况较急，请直接就医。",
                            "truncated": False, "risk_level": "medium",
                            "handled_as": "review_modified", "used_tools": [], "confidence": "low",
                            "retrieved_chunk_ids": [],
                        }
                        continue
                    recorder = st.get("recorder") or TurnRecorder()
                    review = run_review(
                        output,
                        ReviewContext(retrieved_chunk_ids=recorder.retrieved_chunk_ids,
                                      is_closing=False, review_mode="code"),
                        model=None, fix_prompt=load_fix_prompt(), templates=templates,
                        review_mode="code",
                    )
                    results[sid] = {
                        "content": review.output.reply,
                        "truncated": not review.output.reply.strip(),
                        "risk_level": review.output.risk_level,
                        "handled_as": "review_modified" if review.action != "pass" else "normal",
                        "used_tools": recorder.used_tools,
                        "confidence": review.output.assessment.confidence,
                        "retrieved_chunk_ids": sorted(recorder.retrieved_chunk_ids),
                        "review_action": review.action,
                    }

            for item in chunk:
                record = results.get(item["id"]) or {
                    "content": "", "truncated": True, "risk_level": "medium",
                    "handled_as": "review_modified", "used_tools": [], "retrieved_chunk_ids": [],
                }
                fh.write(json.dumps({
                    "id": item["id"], "dimension_code": item["dimension_code"],
                    "task_type": item["task_type"], "layer": "AGENT", "fingerprint": fingerprint,
                    "prompt_version": PROMPT_VERSION, "generation_protocol": GENERATION_PROTOCOL,
                    # 同批共用一次前向，所以记录「摊到每题的秒数」
                    "thinking_mode": "agent-batch",
                    "elapsed_seconds": round((time.time() - t0) / max(len(chunk), 1), 1),
                    **record,
                }, ensure_ascii=False) + "\n")
                written += 1
            fh.flush()
            rate = (time.time() - started) / max(written, 1)
            detail = "、".join(f"{s['round']}×{s['n']}={s['seconds']}s"
                               for s in round_stats[-(len(round_stats)):])
            print(f"[agentb] {written}/{len(pending)} ｜ 本批 {time.time() - t0:.0f}s ｜ "
                  f"平均 {rate:.1f}s/题 ｜ 预计剩余 {rate * (len(pending) - written) / 60:.0f} 分钟\n"
                  f"         各轮耗时：{detail}", flush=True)

    print(f"[agentb] 完成 {written} 题 → {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
