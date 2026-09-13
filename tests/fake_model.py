"""脚本化假模型：离线测试用（不依赖任何模型服务）。

GenericFakeChatModel 不支持 bind_tools，因此这里自写一个最小 BaseChatModel 子类：
按预设脚本依次返回 AIMessage（可含普通工具调用或结构化输出调用）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

STRUCTURED_TOOL_NAME = "AgentTurnOutput"


class ScriptedFakeChatModel(BaseChatModel):
    """按脚本依次返回消息的假模型。

    - responses：AIMessage 列表，每次模型调用弹出一条；
    - calls：记录每次调用收到的消息（供断言）；
    - bound_tools：记录被绑定的工具名。
    """

    responses: list[AIMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake-chat"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001 —— 忽略真实绑定，仅记录
        self.bound_tools = [
            getattr(tool, "name", getattr(tool, "__name__", str(tool))) for tool in tools
        ]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            raise ValueError("ScriptedFakeChatModel：脚本已用尽（收到意料之外的模型调用）")
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])


def structured_call(
    reply: str,
    *,
    assessment: dict | None = None,
    case_card: dict | None = None,
    risk_level: str = "low",
    call_id: str = "call_struct",
) -> AIMessage:
    """构造一次「结构化输出」工具调用。"""
    args = {
        "reply": reply,
        "assessment": assessment if assessment is not None else make_assessment(),
        "case_card": case_card if case_card is not None else make_card(),
        "risk_level": risk_level,
    }
    return AIMessage(
        content="",
        tool_calls=[
            {"name": STRUCTURED_TOOL_NAME, "args": args, "id": call_id, "type": "tool_call"}
        ],
    )


def tool_call(name: str, args: dict, call_id: str = "call_tool") -> AIMessage:
    """构造一次普通工具调用。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def text_response(content: str) -> AIMessage:
    """构造普通文本回复（用于修正重生成路径）。"""
    return AIMessage(content=content)


def make_assessment(
    *,
    confidence: str = "low",
    next_steps: list[str] | None = None,
    hypotheses: list[dict] | None = None,
    missing_info: list[str] | None = None,
    evidence: list[dict] | None = None,
) -> dict:
    return {
        "hypotheses": hypotheses if hypotheses is not None else [],
        "evidence": evidence if evidence is not None else [],
        "missing_info": missing_info if missing_info is not None else [],
        "next_steps": next_steps if next_steps is not None else ["继续观察，如加重请及时就医"],
        "confidence": confidence,
    }


def make_card(**overrides) -> dict:
    card = {
        "age_group": "未提供",
        "sex": "未提供",
        "chief_complaint": "",
        "symptoms": [],
        "timeline": [],
        "context": [],
        "red_flags_found": [],
        "red_flags_excluded": [],
        "hypotheses": [],
        "open_questions": [],
    }
    card.update(overrides)
    return card
