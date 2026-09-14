"""批处理端点 /v1/batch/chat 的逻辑：分组、解析、顺序、预算。

它存在的理由是速度：单条解码 37 tok/s，批解码几百 tok/s，
智能体评测能不能从 2 小时压到 20 分钟全看这里。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pangu_server  # noqa: E402


class _FakeModel:
    """记录每次批量调用的规模与预算，并按预设脚本回内容。"""

    def __init__(self, script: dict[int, str] | None = None) -> None:
        self.calls: list[dict] = []
        self.script = script or {}

    def chat_batch_messages(self, conversations, *, fast_thinking=True, max_new_tokens=1024, **_):
        self.calls.append({"n": len(conversations), "max_new_tokens": max_new_tokens,
                           "fast": fast_thinking, "conv": conversations})
        results = []
        for index, _ in enumerate(conversations):
            text = self.script.get(index, "直接回答患者。")
            results.append({"content": text, "thinking": "", "raw": text,
                            "output_tokens": 7, "fast_thinking": fast_thinking,
                            "stop_reason": "natural", "answered": True})
        return {"results": results, "elapsed_seconds": 0.5,
                "total_output_tokens": 7 * len(conversations), "tokens_per_second": 14.0}


def _tool_request(text_index: int = 0) -> dict:
    return {
        "kind": "tool",
        "messages": [
            {"role": "system", "content": "你是诊断辅助智能体。"},
            {"role": "user", "content": "患者胸痛伴大汗。"},
        ],
        "tools": [{"type": "function", "function": {"name": "check_red_flags",
                                                    "description": "核对危险信号",
                                                    "parameters": {"type": "object", "properties": {}}}}],
        "max_tokens": 320,
        "allowed_tools": ["check_red_flags"],
    }


class TestHandleBatchChat:
    def test_parses_tool_call_and_keeps_order(self):
        model = _FakeModel({0: "TOOL: check_red_flags\nARGS: {\"symptoms\": [\"胸痛\"]}"})
        out = pangu_server.handle_batch_chat(model, {"requests": [_tool_request(), _tool_request()]})
        assert len(out["results"]) == 2
        first = out["results"][0]
        assert first["tool_call"] == {"name": "check_red_flags",
                                      "arguments": {"symptoms": ["胸痛"]}}
        # 第二题脚本回的是纯文本 → 没有工具调用（模型选择直接作答）
        assert out["results"][1]["tool_call"] is None
        assert "直接回答" in out["results"][1]["text"]

    def test_finalize_returns_payload(self):
        payload = ('{"reply": "建议尽快就医。", "assessment": {"hypotheses": [], '
                   '"evidence": [], "missing_info": [], "next_steps": ["就医"], '
                   '"confidence": "low"}, "case_card": {}, "risk_level": "medium"}')
        model = _FakeModel({0: payload})
        out = pangu_server.handle_batch_chat(model, {"requests": [
            {"kind": "finalize", "messages": [{"role": "user", "content": "问题"}],
             "draft_reply": "草稿", "max_tokens": 900}
        ]})
        result = out["results"][0]
        assert result["payload"] is not None
        assert result["payload"]["reply"] == "建议尽快就医。"

    def test_groups_by_budget(self):
        """不同预算的请求必须拆成不同的批，否则短题会被长题的预算拖慢。"""
        model = _FakeModel()
        requests = [_tool_request(), {**_tool_request(), "max_tokens": 1280}]
        pangu_server.handle_batch_chat(model, {"requests": requests})
        assert sorted(c["max_new_tokens"] for c in model.calls) == [320, 1280]
        assert all(c["n"] == 1 for c in model.calls)

    def test_same_budget_goes_into_one_batch(self):
        model = _FakeModel()
        pangu_server.handle_batch_chat(
            model, {"requests": [_tool_request() for _ in range(5)]}
        )
        assert len(model.calls) == 1 and model.calls[0]["n"] == 5

    def test_text_kind_uses_system_and_user(self):
        model = _FakeModel()
        pangu_server.handle_batch_chat(model, {"requests": [
            {"kind": "text", "system": "系统提示", "user": "用户问题", "max_tokens": 512}
        ]})
        conversation = model.calls[0]["conv"][0]
        assert conversation[0] == {"role": "system", "content": "系统提示"}
        assert conversation[1]["content"].startswith("用户问题")

    def test_empty_requests_rejected(self):
        with pytest.raises(ValueError):
            pangu_server.handle_batch_chat(_FakeModel(), {"requests": []})

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError):
            pangu_server.handle_batch_chat(_FakeModel(), {"requests": [{"kind": "???"}]})
