"""审查引擎（一期）：检测 → 修正重生成 → 代码级降级 → 兜底替换。

动作定义（《智能体设计方案.md》§3.5）：
- pass：放行；
- fix：带修正指令重生成一次（附具体违规点）；
- degrade：代码级改写、下调措辞（含红旗一致性强制、剔除伪造引用、补充安全提示）；
- replace：直接替换为模板话术（高危场景，不再经过模型）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from agent.review.rules import ReviewContext, Violation, check_output
from agent.schemas import AgentTurnOutput

FALLBACK_CONTINUE = "抱歉，我需要再确认一下：请告诉我您最主要的不适和持续时间，我来帮您梳理。"

_ADVICE_LINE = "建议尽快到医院就诊评估（如症状加重请立即急诊）。"

_ABSOLUTE_REWRITES = (
    ("绝对不会有事", "可能性较低（但不能完全排除）"),
    ("肯定没事", "暂时没有发现需要紧急处理的信号（但无法完全排除其他可能）"),
    ("绝对没事", "风险看起来较低（但不能完全排除）"),
    ("一定能治好", "配合规范治疗有机会好转"),
    ("一定能好", "配合治疗有机会好转"),
    ("肯定能好", "配合治疗有机会好转"),
    ("保证治好", "无法保证结果，需要规范治疗"),
    ("包治", "需要规范治疗"),
    ("百分之百", "很难做到完全保证"),
)

_CARE_WORDS = ("就医", "就诊", "医院", "面诊", "复查", "急诊", "120")


@dataclass
class ReviewResult:
    """审查结果。action: pass | fix | degrade | replace。"""

    output: AgentTurnOutput
    action: str
    violations: list[Violation]
    notes: list[str]


def _message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _extract_json(text: str) -> dict | None:
    """从模型输出中提取第一个完整的 JSON 对象（容忍代码块围栏与前后杂文）。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start:index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _fix_attempt(model, output: AgentTurnOutput, violations: list[Violation], fix_prompt: str) -> AgentTurnOutput | None:
    payload = json.dumps(output.model_dump(), ensure_ascii=False, indent=2)
    issues = "\n".join(f"- [{violation.code}] {violation.detail}" for violation in violations)
    messages = [
        SystemMessage(content=fix_prompt),
        HumanMessage(content=(
            "【初版输出】\n" + payload + "\n\n【必须修正的问题】\n" + issues +
            "\n\n请输出修正后的完整 JSON（不要使用 markdown 代码块，不要输出任何解释）："
        )),
    ]
    try:
        response = model.invoke(messages)
    except Exception:  # noqa: BLE001 —— 修正失败不抛出，交给兜底处理
        return None
    data = _extract_json(_message_text(response))
    if data is None:
        return None
    try:
        return AgentTurnOutput.model_validate(data)
    except Exception:  # noqa: BLE001
        return None


def _has_care_advice(output: AgentTurnOutput) -> bool:
    advice_text = "；".join(output.assessment.next_steps)
    return any(word in advice_text for word in _CARE_WORDS)


def _apply_degradations(
    output: AgentTurnOutput,
    violations: list[Violation],
    templates: dict[str, str],
) -> tuple[AgentTurnOutput, list[str]]:
    updated = output.model_copy(deep=True)
    notes: list[str] = []
    for violation in violations:
        if violation.code == "red_flag_escalation":
            if updated.risk_level not in ("high", "emergency"):
                updated.risk_level = "high"
            if not _has_care_advice(updated):
                updated.assessment.next_steps.insert(0, _ADVICE_LINE)
            notes.append("已上调风险等级并补充就医建议（危险信号一致性）。")
        elif violation.code == "fabricated_citation":
            bad_refs = set(violation.refs)
            kept = [
                item for item in updated.assessment.evidence
                if not (item.source == "kb" and item.ref in bad_refs)
            ]
            removed = len(updated.assessment.evidence) - len(kept)
            updated.assessment.evidence = kept
            notes.append(f"已剔除 {removed} 条无法验证的知识库引用。")
        elif violation.code == "closing_confidence_cap":
            updated.assessment.confidence = "medium"
            notes.append("已达轮数上限的收口轮次，把握度已下调为 medium。")
        elif violation.code == "closing_advice_missing":
            if not _has_care_advice(updated):
                updated.assessment.next_steps.insert(0, _ADVICE_LINE)
            notes.append("已补充就医建议（中/高把握度评估的必需要求）。")
        elif violation.code == "self_harm_echo":
            suffix = templates.get("self_harm_suffix", "")
            if suffix and suffix not in updated.reply:
                updated.reply = updated.reply.rstrip() + "\n\n" + suffix
            notes.append("已附上安全求助提示。")
    return updated, notes


def _fallback_apply(
    output: AgentTurnOutput,
    violations: list[Violation],
    templates: dict[str, str],
) -> tuple[AgentTurnOutput, list[str]]:
    """修正重生成后仍未解决的问题：执行确定性兜底。"""
    updated = output.model_copy(deep=True)
    actions: list[str] = []
    for violation in violations:
        if violation.code == "absolutist_claim":
            for old, new in _ABSOLUTE_REWRITES:
                updated.reply = updated.reply.replace(old, new)
            actions.append("degrade")
        elif violation.code in ("diagnosis_language", "medication_recommendation"):
            template = templates.get("refuse", "")
            if template:
                updated.reply = template
            if not _has_care_advice(updated):
                updated.assessment.next_steps.insert(0, _ADVICE_LINE)
            actions.append("replace")
        elif violation.code == "fabricated_citation":
            bad_refs = set(violation.refs)
            updated.assessment.evidence = [
                item for item in updated.assessment.evidence
                if not (item.source == "kb" and item.ref in bad_refs)
            ]
            actions.append("degrade")
        elif violation.code == "unsupported_reference":
            updated.reply = re.sub(
                r"(根据|依据|据)(相关|现有的)?(资料|文献|研究|指南)",
                "结合您描述的情况",
                updated.reply,
            )
            actions.append("degrade")
        elif violation.code == "empty_hypotheses":
            updated.assessment.confidence = "low"
            if not _has_care_advice(updated):
                updated.assessment.next_steps.insert(0, _ADVICE_LINE)
            actions.append("degrade")
        elif violation.code == "empty_reply":
            updated.reply = FALLBACK_CONTINUE
            actions.append("replace")
        else:
            actions.append("degrade")
    return updated, actions


def run_review(
    output: AgentTurnOutput,
    ctx: ReviewContext,
    *,
    model=None,
    fix_prompt: str | None = None,
    templates: dict[str, str] | None = None,
    review_mode: str = "code",
) -> ReviewResult:
    """执行完整审查流程，返回最终输出与动作。"""
    templates = templates or {}
    found = check_output(output, ctx)
    if review_mode == "code+llm":
        # 二期模型复核（占位：当前返回 None，不影响流程）
        from agent.review.llm_review import llm_review_pass

        extra = llm_review_pass(output, ctx, model=model, prompt=fix_prompt)
        if extra:
            found.extend(extra)

    actions: list[str] = []
    notes: list[str] = []

    fix_level = [violation for violation in found if violation.action == "fix"]
    if fix_level and model is not None and fix_prompt:
        fixed = _fix_attempt(model, output, fix_level, fix_prompt)
        if fixed is not None:
            output = fixed
            actions.append("fix")
            notes.append("已按审查意见修正重生成（原始问题：" + ", ".join(v.code for v in fix_level) + "）。")
        else:
            notes.append("修正重生成未成功，进入兜底处理。")

    current = check_output(output, ctx)
    degrade_level = [violation for violation in current if violation.action == "degrade"]
    if degrade_level:
        output, new_notes = _apply_degradations(output, degrade_level, templates)
        notes.extend(new_notes)
        actions.append("degrade")

    current = check_output(output, ctx)
    unresolved = [violation for violation in current if violation.action == "fix"]
    if unresolved:
        output, fallback_actions = _fallback_apply(output, unresolved, templates)
        actions.extend(fallback_actions)
        notes.append("以下问题无法通过重生成解决，已执行兜底处理：" + ", ".join(v.code for v in unresolved))

    action = "pass"
    if "replace" in actions:
        action = "replace"
    elif "fix" in actions:
        action = "fix"
    elif "degrade" in actions:
        action = "degrade"

    return ReviewResult(output=output, action=action, violations=found, notes=notes)
