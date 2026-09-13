"""时间线计算工具：把「3天前…」类描述整理为时间线与病程间隔。"""

from __future__ import annotations

import re
from datetime import date, timedelta

from agent.recorder import record_tool_call

_TOOL = "timeline_calc"

_REL = re.compile(r"(\d+)\s*(小时|天|日|周|星期|个月|月|年)\s*前")
_ABS = re.compile(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?")
_WORDS = {
    "今天": 0.0,
    "昨天": 1.0,
    "前天": 2.0,
    "大前天": 3.0,
    "上周": 7.0,
    "上个月": 30.0,
}
_REL_FACTORS = {
    "小时": 1 / 24,
    "天": 1.0,
    "日": 1.0,
    "周": 7.0,
    "星期": 7.0,
    "个月": 30.0,
    "月": 30.0,
    "年": 365.0,
}


def _parse_days_ago(text: str, today: date) -> float | None:
    match = _ABS.search(text)
    if match:
        try:
            parsed = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
        return float((today - parsed).days)
    match = _REL.search(text)
    if match:
        return int(match.group(1)) * _REL_FACTORS[match.group(2)]
    for word, days in _WORDS.items():
        if word in text:
            return days
    return None


def timeline_calc(events: list[str]) -> str:
    """把多个事件按时间排序，并计算各事件距今天数与相邻间隔。

    适用：用户用相对时间描述病程时（如「3天前开始发热」「昨天咳嗽加重」）。
    支持：2026-09-10（或 2026/9/10、2026年9月10日）、3天前、2周前、1个月前、昨天、前天等。
    无法解析的条目会单独列出，请向用户确认具体时间。
    """
    args = {"events": events}
    if not events:
        record_tool_call(_TOOL, args, ok=False, summary="空输入")
        return "错误：事件列表为空。请先向用户收集关键事件与时间，再调用本工具。"

    today = date.today()
    parsed: list[tuple[float, str]] = []
    unparsed: list[str] = []
    for item in events:
        text = str(item).strip()
        if not text:
            continue
        days = _parse_days_ago(text, today)
        if days is None:
            unparsed.append(text)
        else:
            parsed.append((days, text))

    if not parsed:
        record_tool_call(_TOOL, args, ok=False, summary="全部无法解析")
        lines = ["无法解析任何事件的日期。支持格式：2026-09-10、3天前、2周前、1个月前、昨天、前天。"]
        if unparsed:
            lines.append("以下条目需要向用户确认时间：" + "；".join(unparsed))
        return "\n".join(lines)

    parsed.sort(key=lambda pair: -pair[0])  # 距今越久越靠前（由早到晚）
    lines = [f"今天日期：{today.isoformat()}", "时间线（由早到晚）："]
    for index, (days, text) in enumerate(parsed, start=1):
        event_date = today - timedelta(days=round(days))
        if days < 0:
            when = f"日期在未来（≈{event_date.isoformat()}），请与用户确认"
        elif days == 0:
            when = "今天"
        else:
            when = f"约{days:g}天前（≈{event_date.isoformat()}）"
        lines.append(f"{index}. {when}：{text}")
    if len(parsed) >= 2:
        lines.append("相邻事件间隔：")
        for index in range(1, len(parsed)):
            gap = abs(parsed[index][0] - parsed[index - 1][0])
            lines.append(f"- 事件{index} → 事件{index + 1}：约 {gap:g} 天")
    if unparsed:
        lines.append("以下条目无法解析，请向用户确认具体时间：" + "；".join(unparsed))
    record_tool_call(_TOOL, args, ok=True, summary=f"共{len(parsed)}条")
    return "\n".join(lines)
