"""二期模型复核（占位）。

调用点：engine.run_review 在 review_mode == "code+llm" 时调用本模块；
当前返回 None（视为未启用），不产生任何修改。

二期实现内容（见《智能体设计方案.md》§3.5）：
- 证据覆盖：每条主要假设是否有支撑；
- 编造检查：检索为空却声称「有资料/研究表明」；
- 措辞-把握度一致性：confidence=low 却使用确定措辞；
- 收口质量检查。
实现后应返回 list[Violation]（复用 review.rules.Violation）。
"""

from __future__ import annotations


def llm_review_pass(output, ctx, *, model=None, prompt=None):
    """对输出执行语义复核。当前为占位实现：返回 None 表示未启用。"""
    # TODO(二期): 接入模型复核，返回 list[Violation]。
    return None
