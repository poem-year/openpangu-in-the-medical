"""检索接口（A 侧实现落点）与智能体侧检索工具。

分两层（见《智能体接口规范.md》§2.1）：
    - `retrieve()`：本文件是**适配层**，把 `kb.search` 的结果映射成 `Evidence`；
      检索实现、建库、索引都在 `kb/` 包里，这里不写检索逻辑。
    - `retrieve_evidence()`：模型可见的工具，负责降级话术与 chunk_id 上报，保持不变。

注意：`kb` 是惰性导入的——CLI 启动、单元测试都不该因为加载嵌入模型而变慢。
"""

from __future__ import annotations

import json

from agent.recorder import record_tool_call
from agent.schemas import Evidence


def retrieve(query: str, k: int = 5) -> list[Evidence]:
    """检索知识库，返回若干证据片段。

    契约（《智能体接口规范.md》§2）：
    - 正常：返回 ≤k 条，按 score 降序；
    - 无结果：返回空列表 []（不要返回占位/编造证据）；
    - 异常：直接抛出异常，由工具包装层捕获并降级；
    - chunk_id 全局唯一、稳定，绝不编造。

    实现落在 `kb/search.py`；索引缺失时抛 `KbIndexMissingError`。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")

    from kb.config import get_config
    from kb.search import retrieve as kb_retrieve

    config = get_config()
    try:
        top_k = int(k)
    except (TypeError, ValueError):
        top_k = config.top_k
    if top_k < 1:
        top_k = config.top_k
    chunks = kb_retrieve(query, k=top_k, config=config)
    return [
        Evidence(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.source,
            section=chunk.section,
            score=chunk.score,
        )
        for chunk in chunks
    ]


def retrieve_evidence(query: str, k: int = 5) -> str:
    """查询医学资料库，获取与问题相关的资料片段（含真实出处与 chunk_id）。

    适用时机：需要依据资料回答时（如疾病知识、诊疗常规）。
    若返回「未找到」或「不可用」，请如实告知用户，严禁编造资料内容或出处。
    k 为最多返回条数。
    """
    args = {"query": query, "k": k}
    try:
        results = retrieve(query, k=k)
    except Exception as exc:  # noqa: BLE001 —— 契约：任何异常都降级处理，不打断对话
        record_tool_call("retrieve_evidence", args, ok=False, summary=f"异常:{type(exc).__name__}")
        return (
            f"检索服务暂不可用（{type(exc).__name__}）。请如实向用户说明暂时查不到资料，"
            "不要编造任何资料内容或出处。"
        )

    if not results:
        record_tool_call("retrieve_evidence", args, ok=True, summary="空结果")
        return (
            f"未找到与「{query}」相关的资料。请如实告知用户没有查到相关资料（这不代表相关说法不成立），"
            "不要编造资料内容或出处。"
        )

    chunk_ids = [item.chunk_id for item in results]
    record_tool_call("retrieve_evidence", args, ok=True, summary=f"命中{len(chunk_ids)}条", chunk_ids=chunk_ids)
    payload = [
        {
            "chunk_id": item.chunk_id,
            "text": item.text,
            "source": item.source,
            "section": item.section,
            "score": item.score,
        }
        for item in results[:k]
    ]
    return "检索结果（引用时 chunk_id 必须原样使用）：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
