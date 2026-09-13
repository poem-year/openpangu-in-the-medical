"""危险信号工具 + 前置安全扫描（规则数据在 red_flag_rules.py）。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.recorder import record_tool_call
from agent.tools.red_flag_rules import (
    CHILD_PATTERNS,
    EMERGENCY_RULES,
    HIGH_RULES,
    PREGNANCY_PATTERNS,
    PSYCH_PATTERNS,
    SELF_HARM_PATTERNS,
    RedFlagRule,
)

_TOOL = "check_red_flags"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


_NEGATIVE_SPANS = (
    "不疼",
    "不痛",
    "不闷",
    "没疼",
    "没有疼",
    "没有压榨",
    "没有出汗",
    "没出汗",
    "不再",
    "已经不",
    "未出现",
    "没有出现",
)


def _negated(text: str, start: int, end: int) -> bool:
    """简单否定检查：明显否定的表述不触发前置拦截（如「没有压榨感」「不再想死」）。"""
    window = text[max(0, start - 4):end]
    return any(span in window for span in _NEGATIVE_SPANS)


def match_red_flags(text: str) -> list[tuple[RedFlagRule, str]]:
    """返回命中的 (规则, 命中片段) 列表，急症规则在前。"""
    normalized = _normalize(text)
    hits: list[tuple[RedFlagRule, str]] = []
    for rule in EMERGENCY_RULES + HIGH_RULES:
        for pattern in rule.patterns:
            match = re.search(pattern, normalized)
            if match:
                hits.append((rule, match.group(0)))
                break
    return hits


def check_red_flags(symptoms: list[str]) -> str:
    """对照内置危险信号清单，检查症状是否命中「需要紧急/尽快就医」的信号。

    出现任何新症状时调用；传入症状短语列表（如 ["胸痛", "大汗"]）。
    返回命中的条目与建议等级；未命中也**不代表安全**，需结合完整情况判断。
    """
    args = {"symptoms": symptoms}
    if not symptoms:
        record_tool_call(_TOOL, args, ok=False, summary="空输入")
        return "错误：症状列表为空。请先收集症状再核对清单。"

    joined = "；".join(str(item) for item in symptoms if str(item).strip())
    if not joined:
        record_tool_call(_TOOL, args, ok=False, summary="空输入")
        return "错误：症状列表为空。请先收集症状再核对清单。"

    hits = match_red_flags(joined)
    if not hits:
        record_tool_call(_TOOL, args, ok=True, summary="未命中")
        return (
            "未命中危险信号清单中的条目。注意：这不代表安全——清单不完整，需结合完整情况判断；"
            "如症状加重或出现新的不适，请提醒用户及时就医。"
        )

    emergency_hits = [(rule, matched) for rule, matched in hits if rule.level == "emergency"]
    high_hits = [(rule, matched) for rule, matched in hits if rule.level != "emergency"]
    lines: list[str] = []
    if emergency_hits:
        lines.append("【立即急诊 / 拨打 120】命中信号：")
        for rule, matched in emergency_hits:
            lines.append(f"- [{rule.category}] {rule.summary}（命中：「{matched}」）→ 建议：{rule.advice}")
    if high_hits:
        lines.append("【尽快就医】命中信号：")
        for rule, matched in high_hits:
            lines.append(f"- [{rule.category}] {rule.summary}（命中：「{matched}」）→ 建议：{rule.advice}")
    lines.append("请在回复中明确给出上述建议；不要解释「可能是什么病」，不要让用户等待观察。")
    record_tool_call(_TOOL, args, ok=True, summary=f"命中{len(hits)}条")
    return "\n".join(lines)


@dataclass(frozen=True)
class PrescanHit:
    """前置扫描命中项。kind: self_harm | emergency | psych | child | pregnancy。"""

    kind: str
    rule_id: str
    matched: str


def scan_input(text: str, *, first_turn: bool = True) -> PrescanHit | None:
    """前置安全扫描：返回首个命中项；未命中返回 None。

    - 自伤/伤人、急症信号：每一轮都扫描（安全优先）；
    - 超范围场景（精神心理专项 / 儿童 / 孕产专项）：仅在首次对话时识别，
      对话中途出现时不拦截，由提示词要求模型礼貌说明服务范围。
    """
    normalized = _normalize(text)
    if not normalized:
        return None

    for pattern, label in SELF_HARM_PATTERNS:
        match = re.search(pattern, normalized)
        if match and not _negated(normalized, match.start(), match.end()):
            return PrescanHit(kind="self_harm", rule_id=f"self_harm:{label}", matched=match.group(0))

    for rule in EMERGENCY_RULES:
        for pattern in rule.patterns:
            match = re.search(pattern, normalized)
            if match and not _negated(normalized, match.start(), match.end()):
                return PrescanHit(kind="emergency", rule_id=rule.rule_id, matched=match.group(0))

    if first_turn:
        for kind, label, patterns in (
            ("psych", "精神心理专项", PSYCH_PATTERNS),
            ("child", "儿童症状", CHILD_PATTERNS),
            ("pregnancy", "孕产专项", PREGNANCY_PATTERNS),
        ):
            for pattern in patterns:
                match = re.search(pattern, normalized)
                if match and not _negated(normalized, match.start(), match.end()):
                    return PrescanHit(kind=kind, rule_id=f"{kind}:{label}", matched=match.group(0))

    return None
