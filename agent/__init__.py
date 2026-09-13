"""医疗诊断智能体模块（B 线）。

对外入口：answer()（见 agent/agent.py）
数据结构：schemas.py —— 字段与枚举的单一来源，与《智能体接口规范.md》§1 对应。
"""

from agent.errors import AgentError, AgentUnavailableError
from agent.schemas import (
    AgentResult,
    AgentTurnOutput,
    Assessment,
    CaseCard,
    Evidence,
    EvidenceItem,
    Hypothesis,
)

__all__ = [
    "AgentError",
    "AgentUnavailableError",
    "AgentResult",
    "AgentTurnOutput",
    "Assessment",
    "CaseCard",
    "Evidence",
    "EvidenceItem",
    "Hypothesis",
    "answer",
]


def __getattr__(name):
    # 惰性导入 answer，避免导入子模块时触发完整的装配依赖链
    if name == "answer":
        from agent.agent import answer as _answer

        return _answer
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
