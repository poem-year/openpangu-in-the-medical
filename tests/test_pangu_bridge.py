"""服务端桥接层测试：容错解析、协议渲染、结构化输出归一化。

全部离线、不依赖 NPU；用真实模型实测捕获的失败样本做回归。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import pangu_tool_bridge as bridge  # noqa: E402

BUSINESS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_red_flags",
            "description": "对照内置危险信号清单检查症状。补充说明若干。",
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
            "name": "retrieve_evidence",
            "description": "查医学资料。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["query"],
            },
        },
    },
    {"type": "function", "function": {"name": "AgentTurnOutput", "description": "结构化输出", "parameters": {}}},
]


class TestJsonTolerance:
    def test_repairs_truncated_braces(self):
        broken = '{"tool_call": {"name": "check_red_flags", "arguments": {"symptoms": ["咳嗽"]}}'
        assert json.loads(bridge.repair_json(broken))["tool_call"]["name"] == "check_red_flags"

    def test_ignores_trailing_junk(self):
        assert bridge.load_json_tolerant('{"a": 1}}')["a"] == 1
        assert bridge.load_json_tolerant('{"a": 1} 以上。')["a"] == 1

    def test_returns_none_on_prose(self):
        assert bridge.load_json_tolerant("您好，请问咳嗽多久了？") is None


class TestParseToolCall:
    def test_two_line_protocol(self):
        text = 'TOOL: check_red_flags\nARGS: {"symptoms": ["咳嗽", "低热"]}'
        parsed = bridge.parse_tool_call(text, allowed=["check_red_flags"])
        assert parsed is not None
        assert parsed.name == "check_red_flags"
        assert parsed.arguments == {"symptoms": ["咳嗽", "低热"]}

    def test_real_model_truncated_output(self):
        """实测样本：模型少了一个右花括号就停了。"""
        text = '{"tool_call": {"name": "check_red_flags", "arguments": {"symptoms": ["咳嗽", "低热"]}}'
        parsed = bridge.parse_tool_call(text, allowed=["check_red_flags"])
        assert parsed is not None and parsed.arguments["symptoms"] == ["咳嗽", "低热"]

    def test_real_model_extra_brace(self):
        """实测样本：ARGS 后面多了一个 }。"""
        text = 'TOOL: retrieve_evidence\nARGS: {"query": "咳嗽三天 低热 可能原因", "k": 3}}'
        parsed = bridge.parse_tool_call(text, allowed=["retrieve_evidence"])
        assert parsed is not None and parsed.arguments["k"] == 3

    def test_code_fence_is_stripped(self):
        text = '```json\nTOOL: check_red_flags\nARGS: {"symptoms": ["胸痛"]}\n```'
        assert bridge.parse_tool_call(text, allowed=["check_red_flags"]) is not None

    def test_disallowed_tool_is_rejected(self):
        text = 'TOOL: shell\nARGS: {"cmd": "rm -rf /"}'
        assert bridge.parse_tool_call(text, allowed=["check_red_flags"]) is None

    def test_prose_yields_none(self):
        assert bridge.parse_tool_call("请问您的咳嗽有痰吗？", allowed=["check_red_flags"]) is None

    def test_multiple_calls_takes_first_parsable(self):
        """实测样本：慢思考一次吐了两个调用，取第一个合法即可。"""
        text = (
            '{"tool_call": {"name": "check_red_flags", "arguments": {"symptoms": ["咳嗽"]}}}, '
            '{"tool_call": {"name": "timeline_calc", "arguments": {"events": ["3天前"]}}}'
        )
        parsed = bridge.parse_tool_call(text, allowed=["check_red_flags", "timeline_calc"])
        assert parsed is not None and parsed.name == "check_red_flags"

    def test_double_braces_from_prompt_example(self):
        """实测样本：模型照抄示例里的双花括号写法，必须也能解析。"""
        text = 'TOOL: check_red_flags\nARGS: {{"symptoms": ["咳嗽", "低热"]}}'
        parsed = bridge.parse_tool_call(text, allowed=["check_red_flags"])
        assert parsed is not None and parsed.arguments == {"symptoms": ["咳嗽", "低热"]}

    def test_protocol_example_is_valid_json(self):
        """提示词里的示例必须是合法 JSON，不能出现双花括号。"""
        assert '{{' not in bridge.TOOL_PROTOCOL
        example = next(
            line.split("ARGS: ", 1)[1]
            for line in bridge.TOOL_PROTOCOL.splitlines()
            if line.startswith("ARGS: {")
        )
        assert json.loads(example) == {"symptoms": ["胸痛", "大汗"]}


class TestNormalize:
    def test_maps_chinese_enums(self):
        payload = {
            "reply": "您好。",
            "assessment": {
                "hypotheses": [{"name": "上呼吸道感染", "likelihood": "中等"}],
                "evidence": [{"source": "知识库", "ref": "kb::x::1", "detail": "限盐 5 克"}],
                "confidence": "低",
            },
            "risk_level": "急诊",
        }
        out = bridge.normalize_agent_output(payload)
        assert out["assessment"]["hypotheses"][0]["likelihood"] == "medium"
        assert out["assessment"]["evidence"][0]["source"] == "kb"
        assert out["assessment"]["confidence"] == "low"
        assert out["risk_level"] == "emergency"

    def test_fills_required_defaults(self):
        out = bridge.normalize_agent_output({"reply": "先观察。"})
        assert out["assessment"]["next_steps"], "next_steps 不能为空（B 线 schema 要求至少一条）"
        assert out["risk_level"] in {"low", "medium", "high", "emergency"}
        card = out["case_card"]
        assert card["age_group"] == "未提供" and isinstance(card["symptoms"], list)

    def test_uses_draft_when_reply_missing(self):
        out = bridge.normalize_agent_output({}, draft_reply="请问痰是什么颜色？")
        assert out["reply"] == "请问痰是什么颜色？"

    def test_garbage_hypotheses_are_coerced_or_dropped(self):
        """模型常把假设写成字符串：保留并补默认值；没有名字的字典和垃圾直接丢掉。"""
        payload = {
            "reply": "x",
            "assessment": {
                "hypotheses": ["上呼吸道感染", {"likelihood": "high"}, 123, None]
            },
        }
        out = bridge.normalize_agent_output(payload)
        assert [item["name"] for item in out["assessment"]["hypotheses"]] == ["上呼吸道感染"]
        assert out["assessment"]["hypotheses"][0]["likelihood"] == "low"

    def test_fallback_when_nothing_usable(self):
        out = bridge.normalize_agent_output({})
        assert "抱歉" in out["reply"] and out["assessment"]["next_steps"]

    def test_context_and_case_card_hypotheses_shapes(self):
        """case_card.context 是 list[str]、hypotheses 是对象列表（B 线 schema）。"""
        payload = {
            "reply": "您好。",
            "assessment": {"hypotheses": [{"name": "上呼吸道感染", "likelihood": "中"}]},
            "case_card": {"context": "未提供", "hypotheses": ["上呼吸道感染"]},
        }
        out = bridge.normalize_agent_output(payload)
        assert out["case_card"]["context"] == []
        assert out["case_card"]["hypotheses"][0]["name"] == "上呼吸道感染"
        assert out["case_card"]["hypotheses"][0]["likelihood"] == "medium"

    def test_sentinel_values_are_dropped_from_lists(self):
        out = bridge.normalize_agent_output(
            {"reply": "x", "case_card": {"symptoms": ["未提供", "咳嗽"], "timeline": ["未知"]}}
        )
        assert out["case_card"]["symptoms"] == ["咳嗽"]
        assert out["case_card"]["timeline"] == []

    def test_matches_b_line_schema(self):
        """契约测试：归一化结果必须能通过 B 线真实的 Pydantic 校验。

        这条用例是防止「服务端以为对、B 线校验不过」的回归——
        case_card.context / hypotheses 的字段类型就踩过一次。
        """
        from agent.schemas import AgentTurnOutput

        samples = [
            {},
            {"reply": "您好。", "risk_level": "低"},
            {
                "reply": "您好。",
                "assessment": {
                    "hypotheses": ["上呼吸道感染", {"name": "流感", "likelihood": "medium"}],
                    "evidence": [{"source": "对话", "ref": "1", "detail": "咳嗽3天"}],
                    "missing_info": "痰的颜色",
                    "next_steps": "继续观察",
                    "confidence": "中",
                },
                "case_card": {
                    "age_group": "未提供",
                    "symptoms": "咳嗽",
                    "timeline": [],
                    "context": "未提供",
                    "hypotheses": ["上呼吸道感染"],
                    "open_questions": ["痰的颜色？"],
                },
                "risk_level": "medium",
            },
        ]
        for payload in samples:
            normalized = bridge.normalize_agent_output(payload, draft_reply="请问有痰吗？")
            model = AgentTurnOutput.model_validate(normalized)  # 不抛异常即通过
            assert model.reply
            assert model.assessment.next_steps


class TestRendering:
    def test_catalog_is_compact(self):
        catalog = bridge.render_tool_catalog(BUSINESS_TOOLS[:2])
        assert "check_red_flags(symptoms: string[])" in catalog
        assert "retrieve_evidence(query: string, k: integer（可选）)" in catalog
        assert "补充说明" not in catalog, "描述只取第一句，避免提示词过长"

    def test_split_tools(self):
        business, structured = bridge.split_tools(BUSINESS_TOOLS)
        assert [t["function"]["name"] for t in business] == ["check_red_flags", "retrieve_evidence"]
        assert [t["function"]["name"] for t in structured] == ["AgentTurnOutput"]

    def test_tool_system_prompt_contains_protocol(self):
        text = bridge.build_tool_system_prompt("系统提示", BUSINESS_TOOLS[:1], [bridge.FORCE_FINALIZE_NOTE])
        assert "系统提示" in text and "TOOL: <工具名>" in text and "强制要求" in text

    def test_finalize_system_contains_skeleton(self):
        text = bridge.build_finalize_system()
        for field in ("reply", "assessment", "case_card", "risk_level", "open_questions"):
            assert field in text

    def test_transcript_labels_tool_results_with_names(self):
        messages = [
            {"role": "system", "content": "系统"},
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
            {"role": "tool", "tool_call_id": "call_1", "content": "未命中危险信号"}
        ]
        names = bridge.collect_tool_names(messages)
        assert names == {"call_1": "check_red_flags"}
        transcript = bridge.render_transcript(messages, tool_names_by_id=names)
        assert "用户：咳嗽三天" in transcript
        assert "[工具结果] check_red_flags：未命中危险信号" in transcript
        assert "[已调用工具] check_red_flags" in transcript

    def test_count_tool_rounds(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "assistant", "tool_calls": [{"id": "c"}]},
            {"role": "user", "content": "x"},
        ]
        assert bridge.count_tool_rounds(messages) == 3

    def test_finalize_user_prompt_includes_draft(self):
        prompt = bridge.build_finalize_user_prompt("用户：咳嗽", "请问有痰吗？")
        assert "回复草稿" in prompt and "请问有痰吗？" in prompt
