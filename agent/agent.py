"""智能体组装与对外入口。

实现《智能体接口规范.md》§1：
    answer(question, history) -> AgentResult

流程（《智能体设计方案.md》§2.2）：
    前置安全扫描 → 组装消息 → create_agent 主循环 → 结构化产出 → 审查层 → AgentResult
"""

from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.config import Config, get_config
from agent.errors import AgentUnavailableError
from agent.prompt_loader import load_fix_prompt, load_system_prompt, load_templates
from agent.recorder import TurnRecorder, reset_recorder, set_recorder
from agent.review import ReviewContext, run_review
from agent.schemas import AgentResult, AgentTurnOutput, Assessment, EvidenceItem
from agent.tools import ALL_TOOLS
from agent.tools.red_flags import PrescanHit, scan_input

_logger = logging.getLogger("agent")

_FIRST_TURN_NOTE = (
    "【系统指令·首次对话】这是本次会话的第一轮：请在 reply 开头用一句话说明两点——"
    "① 不需要提供姓名、身份证号等身份信息；② 本助手提供初步评估，不能替代医生面诊。"
)

_CLOSING_NOTE = (
    "【系统指令·收口】本次问诊已达到追问轮数上限：本轮必须收口——给出当前评估（目前考虑、主要依据、还缺什么）"
    "与明确的就医建议（next_steps 必须含就医建议）；不要再提出新的追问；把握度不得为 high；"
    "如仍有关键信息缺失，请说明需要就医确认。"
)


def _log(message: str) -> None:
    _logger.info(message)


def build_model(cfg: Config | None = None):
    """构建 OpenAI 兼容模型客户端（《智能体接口规范.md》§3.1）。

    disable_proxy=True（默认）时显式传入 trust_env=False 的 httpx 客户端，
    避免本机代理变量（ALL_PROXY/HTTP_PROXY/HTTPS_PROXY）阻断客户端构建。
    """
    config = cfg or get_config()
    from langchain_openai import ChatOpenAI

    kwargs: dict = {
        "model": config.model_name,
        "temperature": config.temperature,
        "base_url": config.openai_base_url,
        "api_key": config.openai_api_key,
        "timeout": config.request_timeout,
        "max_retries": 2,
    }
    if config.disable_proxy:
        import httpx

        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=config.request_timeout)
    return ChatOpenAI(**kwargs)


def build_agent(*, config: Config | None = None, model=None):
    """组装一期智能体：模型 + 工具 + 中间件 + 结构化输出（response_format）。"""
    cfg = config or get_config()
    chat_model = model or build_model(cfg)
    return create_agent(
        model=chat_model,
        tools=ALL_TOOLS,
        system_prompt=load_system_prompt(),
        middleware=[
            ModelCallLimitMiddleware(run_limit=cfg.model_call_limit),
            ToolCallLimitMiddleware(run_limit=cfg.tool_call_limit),
        ],
        response_format=AgentTurnOutput,
        name="medical_diagnosis_agent",
    )


def _to_messages(history: list[dict] | None) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for index, item in enumerate(history or []):
        if not isinstance(item, dict):
            raise ValueError(f"history[{index}] 必须是 dict（含 role/content 两个键）")
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            messages.append(AIMessage(content=str(content)))
        else:
            raise ValueError(f"history[{index}] 的 role 必须是 'user' 或 'assistant'，收到：{role!r}")
    return messages


def _extract_output(result: dict) -> AgentTurnOutput | None:
    structured = result.get("structured_response")
    if structured is None:
        return None
    if isinstance(structured, AgentTurnOutput):
        return structured
    if isinstance(structured, dict):
        try:
            return AgentTurnOutput.model_validate(structured)
        except Exception:  # noqa: BLE001
            return None
    return None


def _prescan_result(hit: PrescanHit, templates: dict[str, str]) -> AgentResult:
    """前置扫描命中：不经过模型，直接返回固定话术（安全兜底）。"""
    if hit.kind == "self_harm":
        reply = templates.get("self_harm", "")
        next_steps = ["立即联系信任的人陪伴", "拨打心理援助热线 12356 或 120 求助"]
        risk_level = "emergency"
        handled_as = "emergency"
    elif hit.kind == "emergency":
        reply = templates.get("emergency", "")
        next_steps = ["立即拨打 120 或前往最近的急诊科"]
        risk_level = "emergency"
        handled_as = "emergency"
    else:  # psych / child / pregnancy：超范围场景
        reply = templates.get("out_of_scope", "")
        next_steps = ["前往相应专科就诊（儿科 / 妇产科 / 精神心理科）"]
        risk_level = "medium"
        handled_as = "out_of_scope"

    assessment = Assessment(
        hypotheses=[],
        evidence=[EvidenceItem(source="dialogue", ref="本轮输入", detail=f"前置安全扫描命中：{hit.rule_id}")],
        missing_info=[],
        next_steps=next_steps,
        confidence="low",
    )
    return AgentResult(
        reply=reply,
        assessment=assessment,
        risk_level=risk_level,
        used_tools=[],
        handled_as=handled_as,
    )


def answer(
    question: str,
    history: list[dict] | None = None,
    *,
    config: Config | None = None,
    model=None,
    agent=None,
) -> AgentResult:
    """处理单个用户轮次（实现见《智能体接口规范.md》§1.1）。

    - question：用户本轮输入；
    - history：可选，[{"role": "user"|"assistant", "content": "..."}]，按时间顺序；
    - config / model / agent：便于测试与复用（CLI 会预构建 agent+model 后透传）。
    """
    cfg = config or get_config()
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question 不能为空")
    history = history or []
    if not isinstance(history, list):
        raise ValueError("history 必须是 list[dict]")

    templates = load_templates()

    # 1) 前置安全扫描（代码执行，不经过模型）
    hit = scan_input(question, first_turn=not history)
    if hit is not None:
        _log(f"前置扫描命中：{hit.kind} / {hit.rule_id}（{hit.matched}）")
        return _prescan_result(hit, templates)

    # 2) 组装消息（含首次对话提示与收口指令）
    messages = _to_messages(history)
    prior_assistant = sum(
        1 for item in history if isinstance(item, dict) and item.get("role") == "assistant"
    )
    is_closing = prior_assistant >= cfg.max_turns
    if not history:
        messages.append(SystemMessage(content=_FIRST_TURN_NOTE))
    if is_closing:
        messages.append(SystemMessage(content=_CLOSING_NOTE))
    messages.append(HumanMessage(content=question))

    # 3) 运行智能体主循环（工具调用循环由 create_agent 承担）
    runner = agent or build_agent(config=cfg, model=model)
    recorder = TurnRecorder()
    token = set_recorder(recorder)
    try:
        result = runner.invoke({"messages": messages})
    except Exception as exc:  # noqa: BLE001 —— 统一包装为可识别异常
        raise AgentUnavailableError(f"智能体运行失败：{type(exc).__name__}: {exc}") from exc
    finally:
        reset_recorder(token)

    output = _extract_output(result)
    if output is None:
        # 结构化输出缺失（如触发调用上限被安全终止）：返回保守兜底
        _log("未获得结构化输出，返回保守兜底结果。")
        return AgentResult(
            reply=(
                "抱歉，本轮我没能整理出可靠的分析。您可以换一种说法，再描述一次最主要的症状和持续时间；"
                "如果情况较急，请直接就医。"
            ),
            assessment=Assessment(
                hypotheses=[],
                evidence=[EvidenceItem(source="dialogue", ref="本轮输入", detail="智能体未产出结构化输出，已走保守兜底")],
                missing_info=[],
                next_steps=["如症状明显或加重，请及时就医"],
                confidence="low",
            ),
            risk_level="medium",
            used_tools=recorder.used_tools,
            handled_as="review_modified",
        )

    # 4) 审查层（一期硬拦截；REVIEW_MODE=code+llm 时附带二期复核占位）
    ctx = ReviewContext(
        retrieved_chunk_ids=recorder.retrieved_chunk_ids,
        is_closing=is_closing,
        review_mode=cfg.review_mode,
    )
    review = run_review(
        output,
        ctx,
        model=model,
        fix_prompt=load_fix_prompt(),
        templates=templates,
        review_mode=cfg.review_mode,
    )
    if review.violations:
        _log("审查发现：" + "；".join(f"{v.code}({v.action})" for v in review.violations))
    for note in review.notes:
        _log("审查处理：" + note)

    return AgentResult(
        reply=review.output.reply,
        assessment=review.output.assessment,
        risk_level=review.output.risk_level,
        used_tools=recorder.used_tools,
        handled_as="review_modified" if review.action != "pass" else "normal",
    )
