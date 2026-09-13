"""常用单位换算工具（长度 / 重量 / 温度）。

明确不支持药物剂量换算（如 mg↔ml）——遇到此类请求直接说明无法提供。
"""

from __future__ import annotations

from agent.recorder import record_tool_call

_TOOL = "convert_units"

_ALIASES = {
    "°c": "c",
    "℃": "c",
    "摄氏度": "c",
    "摄氏": "c",
    "°f": "f",
    "℉": "f",
    "华氏度": "f",
    "华氏": "f",
}

_LENGTH = {
    "mm": 0.001, "毫米": 0.001,
    "cm": 0.01, "厘米": 0.01,
    "m": 1.0, "米": 1.0,
    "km": 1000.0, "千米": 1000.0,
    "inch": 0.0254, "in": 0.0254, "英寸": 0.0254,
    "ft": 0.3048, "foot": 0.3048, "英尺": 0.3048,
}

_WEIGHT = {
    "kg": 1.0, "千克": 1.0, "公斤": 1.0,
    "g": 0.001, "克": 0.001,
    "斤": 0.5,
    "lb": 0.45359237, "磅": 0.45359237,
}

_VOLUME = {"ml": 1.0, "毫升": 1.0, "l": 1000.0, "升": 1000.0, "cc": 1.0}

_SUPPORTED_TEXT = "当前支持：长度（mm/cm/m/km/英寸/英尺）、重量（kg/克/斤/磅）、温度（摄氏度℃ / 华氏度℉）。"


def _normalize(unit: str) -> str:
    text = unit.strip().lower().replace(" ", "")
    return _ALIASES.get(text, text)


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """换算常用单位（长度 / 重量 / 温度）。

    示例：斤↔千克、厘米↔英寸、摄氏度↔华氏度。
    不支持药物剂量换算；不支持的换算对会返回可用清单。
    """
    args = {"value": value, "from_unit": from_unit, "to_unit": to_unit}
    try:
        val = float(value)
    except (TypeError, ValueError):
        record_tool_call(_TOOL, args, ok=False, summary="非数值输入")
        return "错误：value 必须是数字，请核对后重试。"

    source = _normalize(str(from_unit))
    target = _normalize(str(to_unit))

    if source == target:
        record_tool_call(_TOOL, args, ok=True, summary="单位相同")
        return f"{val:g} {from_unit} = {val:g} {to_unit}（来源与目标单位相同）。"

    if source in ("c", "f") and target in ("c", "f"):
        result = val * 9 / 5 + 32 if source == "c" else (val - 32) * 5 / 9
        record_tool_call(_TOOL, args, ok=True, summary="温度换算")
        return f"{val:g} °{'C' if source == 'c' else 'F'} = {result:.1f} °{'C' if target == 'c' else 'F'}"

    source_is_mass = source in _WEIGHT
    source_is_volume = source in _VOLUME
    target_is_mass = target in _WEIGHT
    target_is_volume = target in _VOLUME
    if (source_is_mass and target_is_volume) or (source_is_volume and target_is_mass):
        record_tool_call(_TOOL, args, ok=False, summary="质量体积互换（剂量场景）")
        return (
            "错误：不支持质量与体积之间的换算（常见于药物剂量场景）。"
            "药物剂量请咨询医生或药师，不要自行换算。"
        )

    for table in (_LENGTH, _WEIGHT):
        if source in table and target in table:
            result = val * table[source] / table[target]
            record_tool_call(_TOOL, args, ok=True, summary="常规换算")
            return f"{val:g} {from_unit} = {result:g} {to_unit}"

    record_tool_call(_TOOL, args, ok=False, summary="不支持的换算对")
    return f"错误：不支持「{from_unit}」→「{to_unit}」的换算。{_SUPPORTED_TEXT}"
