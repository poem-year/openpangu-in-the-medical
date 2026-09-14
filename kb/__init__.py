"""A 线知识库模块：语料接入、向量索引、混合检索、评测辅助。

对外只暴露两件事：
    kb.search.retrieve(query, k)  —— 检索入口，返回 list[RetrievedChunk]
    kb.config.get_config()        —— 配置（环境变量 + 默认值）

智能体侧通过 `agent/retrieval.py` 适配层调用，见 `kb/README.md`。
"""

from __future__ import annotations

from kb.config import KBConfig, get_config

__all__ = ["KBConfig", "get_config"]
