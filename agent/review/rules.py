"""一期硬拦截规则（纯函数，可单测；动作执行在 engine.py）。

设计依据：审查与安全（双层）中的一期部分。
本模块只做检测、返回 Violation 列表，不修改任何内容。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.schemas import AgentTurnOutput


@dataclass(frozen=True)
class Violation:
    """一条审查发现。action: fix=要求模型修正后重生成；degrade=代码级降级处理。"""

    code: str
    action: str
    detail: str
    snippet: str = ""
    refs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ReviewContext:
    """审查所需上下文。"""

    retrieved_chunk_ids: frozenset[str] = frozenset()
    is_closing: bool = False
    review_mode: str = "code"


# ---------------------------------------------------------------- 句式与词表

_GUARD_CHARS = ("不", "没", "未", "别", "勿", "否")
_QUOTE_WORDS = ("说", "问", "提到", "称", "认为", "觉得", "怀疑", "担心", "自称")


def _guarded_before(text: str, index: int, window: int = 6) -> bool:
    """匹配点之前若出现否定词或转述词，则视为误伤（如「不能确诊为」「您说您得了」）。"""
    segment = text[max(0, index - window):index]
    if any(ch in segment for ch in _GUARD_CHARS):
        return True
    return any(word in segment for word in _QUOTE_WORDS)


def _find_guarded(text: str, pattern: str) -> str | None:
    for match in re.finditer(pattern, text):
        if not _guarded_before(text, match.start()):
            return match.group(0)
    return None


DIAGNOSIS_PATTERNS = (
    r"(你|您)得了",
    r"(你|您)(得|患)的是",
    r"(你|您)(就是|肯定是|一定是)(得|患)?",
    r"确诊为",
    r"诊断为",
)

MEDICATION_DRUGS = (
    "布洛芬", "对乙酰氨基酚", "阿司匹林", "阿莫西林", "头孢", "左氧氟沙星", "莫西沙星",
    "甲硝唑", "奥美拉唑", "蒙脱石散", "氯雷他定", "西替利嗪", "硝酸甘油", "速效救心丸",
    "开塞露", "泼尼松", "地塞米松", "强的松", "退烧药", "消炎药", "抗生素", "安眠药",
)

MEDICATION_PATTERNS = (
    r"\d+(\.\d+)?\s*(mg|毫克|微克|μg|ug|g|克|片|粒|ml|毫升|滴|IU|iu|单位)",
    r"(每天|每日|一日|每次|一天)\s*(吃|服|用|喷|滴|喝)\s*\d",
    r"(建议|可以|应该|不妨|试试|吃点|喝点|服用|口服)[^。！？；\n]{0,14}?(" + "|".join(MEDICATION_DRUGS) + ")",
    r"(停(用|掉|药)|不用再吃|先别吃|减量|加量)[^。！？；\n]{0,10}(他汀|降压|降糖|胰岛素|药)",
)

ABSOLUTE_PATTERNS = (
    r"一定能(好|治好)?",
    r"肯定能(好|治好)?",
    r"肯定(没事|没问题)",
    r"绝对(没事|不会有事|没问题)",
    r"包治",
    r"保证(治好|能治好)",
    r"百分之百",
)

SELF_HARM_ECHO_PATTERNS = (
    r"自杀",
    r"轻生",
    r"不想活",
    r"活着没意思",
    r"活不下去",
    r"结束生命",
    r"伤害自己",
    r"自残",
)

UNSUPPORTED_REFERENCE_PATTERN = r"(根据|依据|据)(相关|现有的)?(资料|文献|研究|指南)"

_IMMEDIATE_CARE_WORDS = ("急诊", "120", "急救")
_CARE_WORDS = ("就医", "就诊", "医院", "面诊", "复查", "急诊", "120")


# ---------------------------------------------------------------- 单项检查


def _check_structure(output: AgentTurnOutput) -> list[Violation]:
    violations: list[Violation] = []
    if not output.reply.strip():
        violations.append(Violation(
            code="empty_reply",
            action="fix",
            detail="reply 为空，需要重新生成给患者的回复。",
        ))
    if not output.assessment.hypotheses and output.assessment.confidence != "low":
        violations.append(Violation(
            code="empty_hypotheses",
            action="fix",
            detail="未给出任何候选假设，但把握度不是 low；请补充鉴别方向，或把把握度下调为 low。",
        ))
    return violations


def _check_diagnosis_language(output: AgentTurnOutput) -> list[Violation]:
    for pattern in DIAGNOSIS_PATTERNS:
        matched = _find_guarded(output.reply, pattern)
        if matched:
            return [Violation(
                code="diagnosis_language",
                action="fix",
                detail=f"出现确诊式表述「{matched}」，改为「目前考虑…（需医生面诊确认）」类表述。",
                snippet=matched,
            )]
    return []


def _check_medication(output: AgentTurnOutput) -> list[Violation]:
    for pattern in MEDICATION_PATTERNS:
        matched = _find_guarded(output.reply, pattern)
        if matched:
            return [Violation(
                code="medication_recommendation",
                action="fix",
                detail=f"出现具体用药/剂量表述「{matched}」，请删除并改为建议咨询医生或药师。",
                snippet=matched,
            )]
    return []


def _check_absolutes(output: AgentTurnOutput) -> list[Violation]:
    for pattern in ABSOLUTE_PATTERNS:
        matched = _find_guarded(output.reply, pattern)
        if matched:
            return [Violation(
                code="absolutist_claim",
                action="fix",
                detail=f"出现绝对化承诺「{matched}」，请改为客观、有保留的表述。",
                snippet=matched,
            )]
    return []


def _check_citations(output: AgentTurnOutput, ctx: ReviewContext) -> list[Violation]:
    bad_refs = [
        item.ref
        for item in output.assessment.evidence
        if item.source == "kb" and item.ref not in ctx.retrieved_chunk_ids
    ]
    if bad_refs:
        return [Violation(
            code="fabricated_citation",
            action="degrade",
            detail=f"存在 {len(bad_refs)} 条无法验证的知识库引用（chunk_id 不在本轮检索结果中），将被剔除。",
            refs=tuple(bad_refs),
        )]
    if not ctx.retrieved_chunk_ids:
        matched = re.search(UNSUPPORTED_REFERENCE_PATTERN, output.reply)
        if matched:
            return [Violation(
                code="unsupported_reference",
                action="fix",
                detail=f"回复使用了「{matched.group(0)}」表述，但本轮没有任何检索结果；请改为基于对话信息的表达，或说明未查到资料。",
                snippet=matched.group(0),
            )]
    return []


def _check_red_flag_consistency(output: AgentTurnOutput) -> list[Violation]:
    if not output.case_card.red_flags_found:
        return []
    risk_ok = output.risk_level in ("high", "emergency")
    advice_text = output.reply + "；" + "；".join(output.assessment.next_steps)
    advice_ok = any(word in advice_text for word in _CARE_WORDS + _IMMEDIATE_CARE_WORDS)
    if risk_ok and advice_ok:
        return []
    return [Violation(
        code="red_flag_escalation",
        action="degrade",
        detail="病历卡记录了危险信号，但风险等级或就医建议不足；需上调风险等级并补充就医建议。",
    )]


def _check_closing(output: AgentTurnOutput, ctx: ReviewContext) -> list[Violation]:
    violations: list[Violation] = []
    if ctx.is_closing and output.assessment.confidence == "high":
        violations.append(Violation(
            code="closing_confidence_cap",
            action="degrade",
            detail="已达到追问轮数上限，把握度不得为 high；应下调为 medium。",
        ))
    if output.assessment.confidence in ("medium", "high"):
        advice_text = "；".join(output.assessment.next_steps)
        if not any(word in advice_text for word in _CARE_WORDS):
            violations.append(Violation(
                code="closing_advice_missing",
                action="degrade",
                detail="给出中/高把握度评估时，next_steps 必须包含就医建议；将自动补充。",
            ))
    return violations


def _check_self_harm_echo(output: AgentTurnOutput) -> list[Violation]:
    for pattern in SELF_HARM_ECHO_PATTERNS:
        match = re.search(pattern, output.reply)
        if match and not _guarded_before(output.reply, match.start()):
            return [Violation(
                code="self_harm_echo",
                action="degrade",
                detail="回复中提及自伤相关内容，需附上安全求助提示。",
                snippet=match.group(0),
            )]
    return []


def check_output(output: AgentTurnOutput, ctx: ReviewContext) -> list[Violation]:
    """执行全部一期检查，返回发现列表（可为空）。"""
    violations: list[Violation] = []
    violations.extend(_check_structure(output))
    violations.extend(_check_diagnosis_language(output))
    violations.extend(_check_medication(output))
    violations.extend(_check_absolutes(output))
    violations.extend(_check_citations(output, ctx))
    violations.extend(_check_red_flag_consistency(output))
    violations.extend(_check_closing(output, ctx))
    violations.extend(_check_self_harm_echo(output))
    return violations
