"""单轮工具调用记录：供审查层（引用防伪）与 used_tools 输出使用。

通过 ContextVar 传递：answer() 在调用智能体前设置记录器，调用结束后取回。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    """一次工具调用的记录。"""

    name: str
    args: dict[str, Any]
    ok: bool
    summary: str = ""
    chunk_ids: list[str] = field(default_factory=list)


class TurnRecorder:
    """记录一轮对话内实际发生的工具调用。"""

    def __init__(self) -> None:
        self.records: list[ToolCallRecord] = []

    def record(
        self,
        name: str,
        args: dict[str, Any] | None,
        *,
        ok: bool = True,
        summary: str = "",
        chunk_ids: list[str] | None = None,
    ) -> None:
        self.records.append(
            ToolCallRecord(
                name=name,
                args=dict(args or {}),
                ok=ok,
                summary=summary,
                chunk_ids=list(chunk_ids or []),
            )
        )

    @property
    def used_tools(self) -> list[str]:
        """按调用顺序去重的工具名列表。"""
        seen: list[str] = []
        for item in self.records:
            if item.name not in seen:
                seen.append(item.name)
        return seen

    @property
    def retrieved_chunk_ids(self) -> frozenset[str]:
        """本轮检索工具真实返回过的全部 chunk_id（用于引用防伪）。"""
        ids: set[str] = set()
        for item in self.records:
            ids.update(item.chunk_ids)
        return frozenset(ids)


_current: ContextVar[TurnRecorder | None] = ContextVar("agent_turn_recorder", default=None)


def set_recorder(recorder: TurnRecorder):
    return _current.set(recorder)


def reset_recorder(token) -> None:
    _current.reset(token)


def current_recorder() -> TurnRecorder | None:
    return _current.get()


def record_tool_call(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    ok: bool = True,
    summary: str = "",
    chunk_ids: list[str] | None = None,
) -> None:
    """工具内部调用：把本次调用记录到当前轮次记录器（若存在）。"""
    recorder = _current.get()
    if recorder is not None:
        recorder.record(name, args, ok=ok, summary=summary, chunk_ids=chunk_ids)
