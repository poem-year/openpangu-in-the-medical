"""提示词与固定话术加载（版本化文件在 agent/prompts/ 下）。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_text(filename: str) -> str:
    """读取 prompts 目录下的文本文件。"""
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def load_system_prompt() -> str:
    """主系统提示（v1）。"""
    return load_text("system_prompt_v1.md")


def load_fix_prompt() -> str:
    """审查修正指令（v1）。"""
    return load_text("review_fix_v1.md")


@lru_cache(maxsize=1)
def load_templates() -> dict[str, str]:
    """解析 emergency_templates_v1.md：以「## 名称」开头的段落；文件头说明不参与解析。"""
    text = load_text("emergency_templates_v1.md")
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}
