# -*- coding: utf-8 -*-
"""引用可追溯率（只属于知识库模块的指标，L2/L3/L5 才用）。

定义（与项目口径一致）：

    引用可追溯率 = 结论能在知识库里真实溯源的数量 ÷ 应有依据的结论数量

判定分两级：
1. **可定位**：回答里引用的 chunk_id 确实在检索返回结果里（不是编造、不越界）；
2. **可支撑**：该片段内容确实支持这条结论（机器判不了，交给 AI 判分器 + 人工抽检）。

基线（L0/L1）没有检索、没有引用，本指标按约定固定为 0，不由 D13 的材料引用顶替；
D13 的「材料引用合规率」只作为该维度的自身表现单独报告。

用法（等 A 线检索接上后）：
    from traceability import extract_citations, locatable_rate
    cits = extract_citations(answer)                      # ['abc123', 'def456']
    rate = locatable_rate(cits, retrieved_chunk_ids)      # 可定位率
"""

from __future__ import annotations

import re

# 引用写法：chunk_id: xxx / [chunk_id: xxx] / [来源: xxx] / 裸 chunk_id（8–32 位字母数字）
CHUNK_PATTERNS = [
    re.compile(r"chunk[_\- ]?id\s*[:：]\s*([A-Za-z0-9_\-]{6,40})", re.I),
    re.compile(r"\[\s*([A-Za-z0-9_\-]{6,40})\s*\]"),
    re.compile(r"来源\s*[:：]\s*([A-Za-z0-9_\-]{6,40})"),
]


def extract_citations(answer: str) -> list[str]:
    """从回答里抠出所有引用标识（去重保序）。"""
    found: list[str] = []
    for pattern in CHUNK_PATTERNS:
        for m in pattern.finditer(answer or ""):
            cid = m.group(1).strip()
            if cid not in found:
                found.append(cid)
    return found


def locatable_rate(citations: list[str], valid_chunk_ids) -> float | None:
    """可定位率：引用的 chunk_id 有多少真实存在于检索结果里。"""
    if not citations:
        return None
    valid = {str(c) for c in valid_chunk_ids or []}
    return sum(1 for c in citations if c in valid) / len(citations)


def traceability_rate(answers: list[dict], min_locatable: float = 1.0) -> dict:
    """批量算引用可追溯率。

    answers: [{"id":..., "answer":..., "retrieved_chunk_ids":[...], "supported_flags":[...]}]
    - 分母（默认口径 B）：所有「应有依据」的条目，即给了 retrieved_chunk_ids 的题；
    - 分子：引用全部可定位（且若有 supported_flags，则都被判为可支撑）的条目。
    """
    total = locatable_only = traceable = 0
    per_item = []
    for rec in answers:
        valid = rec.get("retrieved_chunk_ids")
        if not valid:
            continue
        total += 1
        cits = extract_citations(rec.get("answer", ""))
        rate = locatable_rate(cits, valid)
        ok_locatable = bool(rate is not None and rate >= min_locatable)
        supported = rec.get("supported_flags")
        ok_supported = all(supported) if supported else None
        if ok_locatable:
            locatable_only += 1
        if ok_locatable and (ok_supported is not False):
            traceable += 1
        per_item.append({"id": rec.get("id"), "citations": cits,
                         "locatable_rate": rate, "supported": ok_supported})
    return {
        "n": total,
        "locatable_rate": round(locatable_only / total, 4) if total else None,
        "traceability_rate": round(traceable / total, 4) if total else None,
        "items": per_item,
    }
