# -*- coding: utf-8 -*-
"""OpenAI 兼容 HTTP 服务：把 openPangu-R-7B-2512 接到 B 线智能体上。

实现范围（只覆盖 B 线真正用到的能力，见《智能体接口规范.md》§3）：
    GET  /v1/models
    POST /v1/chat/completions   —— 支持 tools / tool_choice / 多轮 tool 消息
不支持：streaming（本期不用）、response_format（结构化输出走 AgentTurnOutput 工具）。

为什么需要这一层：模型没有原生 function calling，而 B 线首轮就发
`tool_choice="required"` + 6 个工具。桥接逻辑（提示词协议、容错解析、收口转换）
在 `scripts/pangu_tool_bridge.py` 里，本文件只负责 HTTP、并发与编排。

启动（见 scripts/serve_pangu.sh）：
    source scripts/pangu_env.sh
    .venv-pangu/bin/python scripts/pangu_server.py --port 8000

调用：
    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 MODEL_NAME=openpangu-7b .venv/bin/python -m agent.cli
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pangu_infer import PanguModel  # noqa: E402
from pangu_tool_bridge import (  # noqa: E402
    FORCE_FINALIZE_NOTE,
    REQUIRE_TOOL_NOTE,
    STRUCTURED_TOOL_NAME,
    build_finalize_system,
    build_finalize_user_prompt,
    build_tool_system_prompt,
    collect_tool_names,
    count_tool_rounds,
    normalize_agent_output,
    parse_json_object,
    parse_tool_call,
    render_transcript,
    split_tools,
)

DEFAULT_MODEL_NAME = "openpangu-7b"


class Settings:
    model_name = DEFAULT_MODEL_NAME
    tool_max_tokens = 320
    finalize_max_tokens = 900
    max_tool_rounds = 3
    require_first_tool = True
    warmup = True
    trace_dir: str | None = "/data/openpangu/logs/pangu_server"


SETTINGS = Settings()


# --------------------------------------------------------------------------
# 消息整形
# --------------------------------------------------------------------------
def collect_system_text(messages: list[dict]) -> str:
    return "\n\n".join(
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "system" and str(message.get("content") or "").strip()
    )


def to_model_messages(messages: list[dict]) -> list[dict]:
    """把 OpenAI 消息转成模型能吃的形式。

    - system 已经被合进工具轮的系统提示，这里丢掉；
    - assistant 的 tool_calls 还原成模型自己写过的 TOOL/ARGS 两行（保持上下文一致）；
    - tool 消息补上工具名（OpenAI 的 tool 消息只有 tool_call_id）。
    """
    names = collect_tool_names(messages)
    rendered: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "assistant":
            calls = message.get("tool_calls") or []
            parts: list[str] = []
            text = message.get("content")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
            for call in calls:
                function = (call or {}).get("function") or {}
                arguments = function.get("arguments")
                try:
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    pass
                parts.append(
                    f"TOOL: {function.get('name')}\nARGS: "
                    f"{json.dumps(arguments, ensure_ascii=False)}"
                )
            rendered.append({"role": "assistant", "content": "\n".join(parts)})
        elif role == "tool":
            name = names.get(str(message.get("tool_call_id") or ""), "工具")
            rendered.append(
                {
                    "role": "tool",
                    "content": f"{name} 返回：{message.get('content') or ''}",
                }
            )
        elif role == "user":
            rendered.append({"role": "user", "content": message.get("content") or ""})
    return rendered


def last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


# --------------------------------------------------------------------------
# 业务编排
# --------------------------------------------------------------------------
def generate_tool_round(
    model: PanguModel,
    messages: list[dict],
    business_tools: list[dict],
    *,
    extra_notes: list[str] | None = None,
) -> dict:
    """工具轮：让模型决定「调用哪个工具」或「直接回答」。"""
    system_text = collect_system_text(messages)
    rounds = sum(len(message.get("tool_calls") or []) for message in messages)
    notes = [FORCE_FINALIZE_NOTE] if rounds >= SETTINGS.max_tool_rounds else []
    notes.extend(extra_notes or [])
    system_prompt = build_tool_system_prompt(system_text, business_tools, notes)
    payload = [{"role": "system", "content": system_prompt}] + to_model_messages(messages)
    return model.chat_messages(
        payload, fast_thinking=True, max_new_tokens=SETTINGS.tool_max_tokens
    )


def generate_finalize_round(
    model: PanguModel, messages: list[dict], draft_reply: str = "", *, retry_note: str = ""
) -> tuple[dict[str, Any] | None, dict]:
    """收口轮：把对话与工具结果转成一个 AgentTurnOutput JSON。"""
    names = collect_tool_names(messages)
    transcript = render_transcript(messages, tool_names_by_id=names)
    system_prompt = build_finalize_system()
    if retry_note:
        system_prompt = f"{system_prompt}\n\n{retry_note}"
    user_prompt = build_finalize_user_prompt(transcript, draft_reply)
    out = model.chat_messages(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        fast_thinking=True,
        max_new_tokens=SETTINGS.finalize_max_tokens,
    )
    payload = parse_json_object(out["content"])
    return payload, out


def build_batch_conversation(model: PanguModel, request: dict) -> tuple[list[dict], int]:
    """把一条批量请求转成 (模型消息列表, 输出预算)。

    三种 kind：
    - `tool`：工具轮，把业务工具渲染进系统提示，要求模型输出 TOOL/ARGS 或直接作答；
      `notes` 可追加强制说明（如首轮没调工具时的补偿提示）。
    - `finalize`：收口轮，把对话渲染成 transcript 后要一个 AgentTurnOutput JSON。
    - `text`：直接给 system + user 的文本轮（服务端不需要解析结构）。
    """
    kind = str(request.get("kind") or "tool")
    messages = request.get("messages") or []
    max_tokens = int(request.get("max_tokens") or 0)

    if kind == "tool":
        business_tools = [tool for tool in (request.get("tools") or []) if tool]
        system_prompt = build_tool_system_prompt(
            collect_system_text(messages), business_tools, list(request.get("notes") or [])
        )
        return [{"role": "system", "content": system_prompt}] + to_model_messages(messages), (
            max_tokens or SETTINGS.tool_max_tokens
        )

    if kind == "finalize":
        names = collect_tool_names(messages)
        transcript = render_transcript(messages, tool_names_by_id=names)
        system_prompt = build_finalize_system()
        if request.get("retry_note"):
            system_prompt = f"{system_prompt}\n\n{request['retry_note']}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": build_finalize_user_prompt(transcript, request.get("draft_reply") or "")},
        ], (max_tokens or SETTINGS.finalize_max_tokens)

    if kind == "text":
        conversation = []
        if request.get("system"):
            conversation.append({"role": "system", "content": str(request["system"])})
        conversation.append({"role": "user", "content": str(request.get("user") or "")})
        return conversation, (max_tokens or SETTINGS.finalize_max_tokens)

    raise ValueError(f"不支持的 kind：{kind!r}（只能是 tool / finalize / text）")


def handle_batch_chat(model: PanguModel, body: dict) -> dict:
    """批处理端点：一次把**同一轮**的 N 道题打包并行解码。

    请求体：`{"requests": [{"kind": "tool"|"finalize"|"text", ...}, ...]}`
    响应体：`{"results": [...]}`，顺序与请求一致；每条含
    `text`（原始输出）、`tool_call`（kind=tool 时解析出来的工具调用，可能为 null）、
    `payload`（kind=finalize 时归一化后的 AgentTurnOutput 字典，可能为 null）、
    `input_tokens` / `output_tokens`。

    内部按 (输出预算, 快思考) 分组，每组一次 `chat_batch_messages`——
    这是把智能体评测从 37 tok/s 提到几百 tok/s 的关键。
    """
    started = time.time()
    requests = body.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests 必须是非空数组")

    prepared: list[dict] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"requests[{index}] 必须是对象")
        conversation, max_tokens = build_batch_conversation(model, request)
        prepared.append({
            "index": index,
            "kind": str(request.get("kind") or "tool"),
            "conversation": conversation,
            "max_tokens": max_tokens,
            "fast_thinking": bool(request.get("fast_thinking", True)),
            "allowed_tools": [t for t in (request.get("allowed_tools") or []) if t],
        })

    results: list[dict | None] = [None] * len(prepared)
    groups: dict[tuple[int, bool], list[dict]] = {}
    for item in prepared:
        groups.setdefault((item["max_tokens"], item["fast_thinking"]), []).append(item)

    total_out = 0
    for (max_tokens, fast_thinking), items in groups.items():
        out = model.chat_batch_messages(
            [item["conversation"] for item in items],
            fast_thinking=fast_thinking,
            max_new_tokens=max_tokens,
        )
        total_out += int(out["total_output_tokens"])
        for item, res in zip(items, out["results"]):
            text = res.get("content") or ""
            record = {
                "kind": item["kind"],
                "text": text,
                "thinking": res.get("thinking") or "",
                "output_tokens": int(res.get("output_tokens") or 0),
                "stop_reason": res.get("stop_reason", "natural"),
                "tool_call": None,
                "payload": None,
            }
            if item["kind"] == "tool":
                parsed = parse_tool_call(text, allowed=item["allowed_tools"] or None)
                if parsed is not None:
                    record["tool_call"] = {"name": parsed.name, "arguments": parsed.arguments}
            elif item["kind"] == "finalize":
                record["payload"] = normalize_agent_output(
                    parse_json_object(text) or {}, draft_reply=""
                )
            results[item["index"]] = record

    elapsed = time.time() - started
    return {
        "results": results,
        "groups": len(groups),
        "elapsed_seconds": round(elapsed, 2),
        "total_output_tokens": total_out,
        "tokens_per_second": round(total_out / max(elapsed, 1e-6), 1),
    }


def handle_chat_completion(model: PanguModel, body: dict) -> dict:
    """一次 /v1/chat/completions 的全部编排。"""
    started = time.time()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空数组")
    if body.get("stream"):
        raise ValueError("本服务不支持 streaming（stream=true），请改为 stream=false")
    tools = body.get("tools") or []
    tool_choice = body.get("tool_choice", "auto")
    business_tools, structured_tools = split_tools(tools)
    business_names = [
        ((tool or {}).get("function") or {}).get("name") for tool in business_tools
    ]
    forced_name = ""
    if isinstance(tool_choice, dict):
        forced_name = str(((tool_choice.get("function") or {}).get("name")) or "")

    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def add_usage(out: dict) -> None:
        usage["prompt_tokens"] += int(out.get("input_tokens") or 0)
        usage["completion_tokens"] += int(out.get("output_tokens") or 0)
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    # tool_choice="none"：不调工具，直接给文本
    if tool_choice == "none":
        out = model.chat_messages(
            [{"role": "system", "content": collect_system_text(messages)}] + to_model_messages(messages),
            fast_thinking=True,
            max_new_tokens=SETTINGS.finalize_max_tokens,
        )
        add_usage(out)
        return _completion(body, _text_message(out["content"]), "stop", usage, started)

    # 明确要求调结构化输出工具 → 直接收口
    if forced_name == STRUCTURED_TOOL_NAME:
        payload, out = generate_finalize_round(model, messages)
        add_usage(out)
        normalized = normalize_agent_output(payload or {})
        return _completion(
            body, _tool_message(STRUCTURED_TOOL_NAME, normalized), "tool_calls", usage, started
        )

    # 工具轮
    out = generate_tool_round(model, messages, business_tools)
    add_usage(out)
    allowed = [name for name in business_names if name]
    parsed = parse_tool_call(out["content"], allowed=allowed) if allowed else None
    if parsed is not None:
        return _completion(
            body,
            _tool_message(parsed.name, parsed.arguments),
            "tool_calls",
            usage,
            started,
            note=f"tool:{parsed.name}",
        )

    # 补偿：模型第一轮就抢答、一个工具都没调时，带着强制说明再问一轮。
    # 7B 对「出现症状先核对危险信号」这类规则遵守得不好，这一步把它拉回 agentic 流程。
    if SETTINGS.require_first_tool and allowed and not any(
        message.get("role") == "assistant" and message.get("tool_calls") for message in messages
    ):
        prompted = generate_tool_round(
            model, messages, business_tools, extra_notes=[REQUIRE_TOOL_NOTE]
        )
        add_usage(prompted)
        forced = parse_tool_call(prompted["content"], allowed=allowed)
        if forced is not None:
            return _completion(
                body,
                _tool_message(forced.name, forced.arguments),
                "tool_calls",
                usage,
                started,
                note=f"tool:{forced.name}(nudged)",
            )
        out = prompted

    # 请求里没有结构化输出工具（不是 B 线，而是普通 OpenAI 兼容调用）：
    # 直接把它的话当回答返回，不要伪造一个 AgentTurnOutput。
    if not structured_tools:
        draft = (out["content"] or "").strip()
        return _completion(
            body, _text_message(draft), "stop", usage, started, note="text(no-structured-tool)"
        )

    # 模型没走协议：把它当「可以直接回答了」→ 收口成结构化输出
    draft = (out["content"] or "").strip()
    payload, fin_out = generate_finalize_round(model, messages, draft)
    add_usage(fin_out)
    if payload is None:
        payload, fin_out = generate_finalize_round(
            model,
            messages,
            draft,
            retry_note=(
                "上一次输出不是合法 JSON。这次务必只输出一个 JSON 对象，"
                "第一个字符是 {，不要输出解释、不要复读模板。"
            ),
        )
        add_usage(fin_out)
    if payload is None:
        # 两次都失败：返回纯文本，交给 B 线的兜底逻辑处理，不替模型编造结构
        return _completion(
            body, _text_message(draft), "stop", usage, started, note="finalize_failed"
        )
    normalized = normalize_agent_output(payload, draft_reply=draft)
    return _completion(
        body,
        _tool_message(STRUCTURED_TOOL_NAME, normalized),
        "tool_calls",
        usage,
        started,
        note="finalize",
    )


# --------------------------------------------------------------------------
# OpenAI 响应
# --------------------------------------------------------------------------
def _text_message(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _tool_message(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _completion(
    body: dict,
    message: dict,
    finish_reason: str,
    usage: dict,
    started: float,
    *,
    note: str = "",
) -> dict:
    elapsed = round(time.time() - started, 2)
    if SETTINGS.trace_dir:
        _trace(body, message, finish_reason, usage, elapsed, note)
    if note:
        print(f"[req] {note} 用时 {elapsed}s 输出 {usage['completion_tokens']} tok", flush=True)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or SETTINGS.model_name,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": usage,
    }


def _trace(body: dict, message: dict, finish_reason: str, usage: dict, elapsed: float, note: str) -> None:
    """把每次请求的关键信息落盘（便于事后复盘提示词与解析失败）。"""
    try:
        os.makedirs(SETTINGS.trace_dir, exist_ok=True)
        path = os.path.join(SETTINGS.trace_dir, time.strftime("%Y-%m-%d") + ".jsonl")
        record = {
            "time": time.strftime("%H:%M:%S"),
            "elapsed": elapsed,
            "note": note,
            "finish_reason": finish_reason,
            "usage": usage,
            "question": last_user_text(body.get("messages") or []),
            "message": message,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    model: PanguModel
    lock: threading.Lock

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的命名
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": SETTINGS.model_name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "openpangu",
                        }
                    ],
                },
            )
        elif self.path.rstrip("/") in ("/health", "/healthz"):
            self._send_json(200, {"status": "ok", "model": SETTINGS.model_name})
        else:
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions", "/v1/batch/chat", "/batch/chat"):
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(
                400, {"error": {"message": f"请求体不是合法 JSON：{exc}", "type": "invalid_request_error"}}
            )
            return
        try:
            with self.lock:
                if path in ("/v1/batch/chat", "/batch/chat"):
                    response = handle_batch_chat(self.model, body)
                else:
                    response = handle_chat_completion(self.model, body)
        except ValueError as exc:
            self._send_json(
                400, {"error": {"message": str(exc), "type": "invalid_request_error"}}
            )
            return
        except Exception as exc:  # noqa: BLE001 - 任何生成异常都要变成可读的错误体
            print(f"[error] {type(exc).__name__}: {exc}", flush=True)
            self._send_json(
                500, {"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}}
            )
            return
        self._send_json(200, response)

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - 静默默认访问日志
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="openPangu OpenAI 兼容服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tool-max-tokens", type=int, default=Settings.tool_max_tokens)
    parser.add_argument("--finalize-max-tokens", type=int, default=Settings.finalize_max_tokens)
    parser.add_argument("--max-tool-rounds", type=int, default=Settings.max_tool_rounds,
                        help="工具轮数上限，超过就强制收口（控制延迟）")
    parser.add_argument("--no-require-first-tool", action="store_true",
                        help="关掉「第一轮必须调一次工具」的补偿（纯模型行为，L4 消融时用）")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热（首次请求会慢 5~7 秒）")
    parser.add_argument("--trace-dir", default=Settings.trace_dir, help="请求留痕目录（空字符串关闭）")
    args = parser.parse_args()

    SETTINGS.model_name = args.model_name
    SETTINGS.tool_max_tokens = args.tool_max_tokens
    SETTINGS.finalize_max_tokens = args.finalize_max_tokens
    SETTINGS.max_tool_rounds = args.max_tool_rounds
    SETTINGS.require_first_tool = not args.no_require_first_tool
    SETTINGS.trace_dir = args.trace_dir or None

    model = PanguModel()
    print(f"[start] 加载模型 {model.model_path} …", flush=True)
    model.load()
    if not args.no_warmup:
        t0 = time.time()
        model.chat("预热", system_prompt=None, fast_thinking=True, max_new_tokens=4)
        print(f"[start] 预热完成 {time.time() - t0:.1f}s（首次调用要编译昇腾算子）", flush=True)

    Handler.model = model
    Handler.lock = threading.Lock()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[start] 服务就绪：http://{args.host}:{args.port}/v1　模型名 {SETTINGS.model_name}\n"
        f"        B 线用法：OPENAI_BASE_URL=http://{args.host}:{args.port}/v1 "
        f"MODEL_NAME={SETTINGS.model_name} .venv/bin/python -m agent.cli",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[stop] 收到中断，正在退出", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
