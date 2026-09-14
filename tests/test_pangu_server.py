"""服务端编排测试：用假模型跑通工具轮 / 收口 / 兜底分支（不加载 NPU）。

真模型端到端一次要十几到几十秒，这里用脚本化假模型把编排逻辑钉死；
模型的真实行为由 scripts/smoke_agent_server.py 做冒烟验证。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import pangu_server as server  # noqa: E402

STRUCTURED = {
    "reply": "您好，先观察。",
    "assessment": {
        "hypotheses": ["上呼吸道感染"],
        "evidence": [{"source": "dialogue", "ref": "1", "detail": "咳嗽3天"}],
        "missing_info": ["痰的颜色"],
        "next_steps": ["继续观察体温"],
        "confidence": "low",
    },
    "case_card": {"symptoms": ["咳嗽3天"], "context": "未提供"},
    "risk_level": "low",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_red_flags",
            "description": "危险信号核查。",
            "parameters": {
                "type": "object",
                "properties": {"symptoms": {"type": "array", "items": {"type": "string"}}},
                "required": ["symptoms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "AgentTurnOutput",
            "description": "结构化输出",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class FakeModel:
    """按脚本依次返回内容；记录每次收到的消息，便于断言提示词。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat_messages(self, messages, **_kwargs):
        self.calls.append([dict(item) for item in messages])
        content = self.replies.pop(0) if self.replies else ""
        return {
            "content": content,
            "thinking": "",
            "raw": content,
            "input_tokens": 10,
            "output_tokens": max(1, len(content) // 4),
            "elapsed_seconds": 0.0,
            "tokens_per_second": 0.0,
            "fast_thinking": True,
        }


def _body(*messages, tools=None, tool_choice="required"):
    return {
        "model": "openpangu-7b",
        "messages": list(messages),
        "tools": TOOLS if tools is None else tools,
        "tool_choice": tool_choice,
        "temperature": 0.1,
        "stream": False,
    }


@pytest.fixture(autouse=True)
def _no_trace(monkeypatch):
    monkeypatch.setattr(server.SETTINGS, "trace_dir", None)
    # 默认关掉「第一轮必须调工具」的补偿：多数用例只关心工具轮/收口本身。
    # 需要验证补偿的用例自己打开（见 TestToolRound）。
    monkeypatch.setattr(server.SETTINGS, "require_first_tool", False)


class TestToolRound:
    def test_returns_business_tool_call(self):
        model = FakeModel(['TOOL: check_red_flags\nARGS: {"symptoms": ["咳嗽", "低热"]}'])
        body = _body({"role": "system", "content": "系统提示"}, {"role": "user", "content": "咳嗽三天"})
        response = server.handle_chat_completion(model, body)
        choice = response["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "check_red_flags"
        assert json.loads(call["function"]["arguments"]) == {"symptoms": ["咳嗽", "低热"]}
        assert call["id"].startswith("call_")

    def test_prompt_contains_system_and_tool_catalog(self):
        model = FakeModel(['TOOL: check_red_flags\nARGS: {"symptoms": ["咳嗽"]}'])
        server.handle_chat_completion(
            model, _body({"role": "system", "content": "B 线系统提示"}, {"role": "user", "content": "咳嗽"})
        )
        system_prompt = model.calls[0][0]["content"]
        assert "B 线系统提示" in system_prompt
        assert "check_red_flags(symptoms: string[])" in system_prompt
        assert "AgentTurnOutput" not in system_prompt, "结构化输出工具不该出现在工具轮目录里"

    def test_tool_history_is_rendered_for_model(self):
        model = FakeModel(['TOOL: check_red_flags\nARGS: {"symptoms": ["咳嗽"]}'])
        messages = [
            {"role": "user", "content": "咳嗽三天"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "check_red_flags", "arguments": '{"symptoms": ["咳嗽"]}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "未命中危险信号"},
        ]
        server.handle_chat_completion(model, _body(*messages))
        payload = model.calls[0]
        assistant = [m for m in payload if m["role"] == "assistant"][0]
        assert "TOOL: check_red_flags" in assistant["content"]
        tool_message = [m for m in payload if m["role"] == "tool"][0]
        assert "check_red_flags 返回：未命中危险信号" in tool_message["content"]

    def test_force_finalize_note_after_round_limit(self):
        model = FakeModel(['TOOL: check_red_flags\nARGS: {"symptoms": ["x"]}'])
        messages = [
            {"role": "user", "content": "咳嗽"},
        ]
        for index in range(server.SETTINGS.max_tool_rounds):
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": "check_red_flags", "arguments": "{}"},
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": f"call_{index}", "content": "ok"})
        server.handle_chat_completion(model, _body(*messages))
        assert "强制要求" in model.calls[0][0]["content"]

    def test_nudges_model_into_calling_a_tool_first(self):
        """模型第一轮抢答时，服务端再问一轮把它拉回工具调用。"""
        server.SETTINGS.require_first_tool = True
        model = FakeModel(
            ["请问您的咳嗽有痰吗？", 'TOOL: check_red_flags\nARGS: {"symptoms": ["咳嗽"]}']
        )
        response = server.handle_chat_completion(
            model, _body({"role": "user", "content": "咳嗽三天"})
        )
        call = response["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "check_red_flags"
        assert len(model.calls) == 2
        assert "本轮强制要求" in model.calls[1][0]["content"]

    def test_nudge_is_skipped_when_tool_already_used(self):
        """已经有工具调用记录时不再补问，避免每轮都多跑一次。"""
        server.SETTINGS.require_first_tool = True
        model = FakeModel(["请问有痰吗？", json.dumps(STRUCTURED, ensure_ascii=False)])
        messages = [
            {"role": "user", "content": "咳嗽"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "check_red_flags", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "未命中"},
        ]
        response = server.handle_chat_completion(model, _body(*messages))
        assert response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "AgentTurnOutput"
        assert len(model.calls) == 2, "一轮工具轮 + 一轮收口，不再补问"


class TestFinalize:
    def test_prose_triggers_finalize(self):
        model = FakeModel(["请问您的咳嗽有痰吗？", json.dumps(STRUCTURED, ensure_ascii=False)])
        response = server.handle_chat_completion(
            model, _body({"role": "user", "content": "咳嗽三天"})
        )
        choice = response["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "AgentTurnOutput"
        payload = json.loads(call["function"]["arguments"])
        assert payload["reply"] == "您好，先观察。"
        assert len(model.calls) == 2, "工具轮 + 收口轮各一次"
        assert "回复草稿" in model.calls[1][-1]["content"]

    def test_broken_json_is_repaired(self):
        """实测失败模式：模型把结尾的 } 少写了，必须能补回来。"""
        truncated = json.dumps(STRUCTURED, ensure_ascii=False)[:-2]
        model = FakeModel(["请问有痰吗？", truncated])
        response = server.handle_chat_completion(model, _body({"role": "user", "content": "咳嗽"}))
        call = response["choices"][0]["message"]["tool_calls"][0]
        assert json.loads(call["function"]["arguments"])["reply"] == "您好，先观察。"

    def test_truncated_mid_key_drops_half_written_pair(self):
        """截断在「键名后没写值」时，丢掉这半截键值对再闭合，不必多跑一轮。"""
        broken = json.dumps(STRUCTURED, ensure_ascii=False)[:-30]
        model = FakeModel(["请问有痰吗？", broken])
        response = server.handle_chat_completion(model, _body({"role": "user", "content": "咳嗽"}))
        call = response["choices"][0]["message"]["tool_calls"][0]
        payload = json.loads(call["function"]["arguments"])
        assert payload["reply"] == "您好，先观察。"
        assert payload["case_card"]["context"] == [], "被截断的字段走默认值"
        assert len(model.calls) == 2, "工具轮 + 收口（修复成功，无需重试）"

    def test_finalize_retry_then_text_fallback(self):
        model = FakeModel(
            ["请问有痰吗？", "抱歉，我不太确定。", "还是不行。"]
        )
        response = server.handle_chat_completion(model, _body({"role": "user", "content": "咳嗽"}))
        choice = response["choices"][0]
        assert choice["finish_reason"] == "stop", "两次收口都失败时交回纯文本，由 B 线兜底"
        assert choice["message"]["content"] == "请问有痰吗？"
        assert len(model.calls) == 3

    def test_forced_structured_tool_skips_tool_round(self):
        model = FakeModel([json.dumps(STRUCTURED, ensure_ascii=False)])
        body = _body(
            {"role": "user", "content": "咳嗽"},
            tool_choice={"type": "function", "function": {"name": "AgentTurnOutput"}},
        )
        response = server.handle_chat_completion(model, body)
        assert response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "AgentTurnOutput"
        assert len(model.calls) == 1

    def test_tool_choice_none_returns_text(self):
        model = FakeModel(["您好，请多休息。"])
        response = server.handle_chat_completion(
            model, _body({"role": "user", "content": "咳嗽"}, tool_choice="none")
        )
        choice = response["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == "您好，请多休息。"


class TestUsageAndErrors:
    def test_usage_is_accumulated_across_rounds(self):
        model = FakeModel(["请问有痰吗？", json.dumps(STRUCTURED, ensure_ascii=False)])
        response = server.handle_chat_completion(model, _body({"role": "user", "content": "咳嗽"}))
        usage = response["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"] > 0

    def test_empty_messages_raises(self):
        with pytest.raises(ValueError):
            server.handle_chat_completion(FakeModel([]), _body())

    def test_plain_chat_without_tools_returns_text(self):
        """回归：不带 tools 的普通请求不能返回 AgentTurnOutput（语义错且白跑一轮收口）。"""
        model = FakeModel(["您好，请问有什么可以帮您？", json.dumps(STRUCTURED, ensure_ascii=False)])
        body = {"model": "x", "messages": [{"role": "user", "content": "你好"}]}
        response = server.handle_chat_completion(model, body)
        choice = response["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"].get("tool_calls") is None
        assert choice["message"]["content"] == "您好，请问有什么可以帮您？"
        assert len(model.calls) == 1, "不该再多跑一次收口生成"

    def test_tools_without_structured_tool_returns_text(self):
        """只给了业务工具（没给 AgentTurnOutput）时同样返回文本。"""
        model = FakeModel(["请多休息。"])
        body = _body(
            {"role": "user", "content": "咳嗽"},
            tools=[TOOLS[0]],
        )
        response = server.handle_chat_completion(model, body)
        choice = response["choices"][0]
        assert choice["finish_reason"] == "stop" and choice["message"]["content"] == "请多休息。"

    def test_streaming_is_rejected(self):
        body = _body({"role": "user", "content": "咳嗽"})
        body["stream"] = True
        with pytest.raises(ValueError, match="streaming"):
            server.handle_chat_completion(FakeModel([]), body)
