"""BMI 计算工具（仅成人）。"""

from __future__ import annotations

from agent.recorder import record_tool_call

_TOOL = "calculate_bmi"


def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """计算成人 BMI 并给出中国成人标准分级。

    适用：用户提供了体重（千克）与身高（厘米）时的客观计算。
    不适用：儿童与孕产专项人群（超出服务范围）。
    数值明显超出合理范围时会返回错误提示，请与用户核对后重试。
    """
    args = {"weight_kg": weight_kg, "height_cm": height_cm}
    try:
        weight = float(weight_kg)
        height = float(height_cm)
    except (TypeError, ValueError):
        record_tool_call(_TOOL, args, ok=False, summary="非数值输入")
        return "错误：体重与身高必须是数字（体重单位千克、身高单位厘米），请核对后重试。"

    if not 2 <= weight <= 500:
        record_tool_call(_TOOL, args, ok=False, summary="体重超出范围")
        return "错误：体重数值超出合理范围（2~500 千克）。请先与用户核对数值与单位，不要基于该数值继续计算。"
    if not 100 <= height <= 250:
        record_tool_call(_TOOL, args, ok=False, summary="身高超出范围")
        return "错误：身高数值超出合理范围（100~250 厘米）。请先与用户核对数值与单位，不要基于该数值继续计算。"

    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        category = "偏瘦"
    elif bmi < 24:
        category = "正常"
    elif bmi < 28:
        category = "超重"
    else:
        category = "肥胖"

    record_tool_call(_TOOL, args, ok=True, summary=f"BMI={bmi:.1f}（{category}）")
    return (
        f"BMI = {bmi:.1f}，分级：{category}"
        "（依据中国成人标准：<18.5 偏瘦；18.5~23.9 正常；24~27.9 超重；≥28 肥胖）。"
        "注意：仅适用于成人；该分级不能替代整体医学评估。"
    )
