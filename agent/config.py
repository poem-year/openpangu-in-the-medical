"""配置读取：环境变量 + 项目根目录 .env（零额外依赖的极简解析）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """智能体运行配置（默认值见《环境说明.md》§六）。"""

    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = "not-needed"
    model_name: str = "openpangu-7b"
    max_turns: int = 6
    temperature: float = 0.1
    review_mode: str = "code"
    request_timeout: float = 60.0
    debug: bool = False
    disable_proxy: bool = True
    model_call_limit: int = 8
    tool_call_limit: int = 10


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


_env_loaded = False


def _find_env_file() -> Path | None:
    cwd = Path.cwd()
    candidates = [
        cwd / ".env",
        cwd.parent / ".env",
        cwd.parent.parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_env_file(path: Path | None = None) -> Path | None:
    """读取 .env（KEY=VALUE，# 开头为注释）；不覆盖已存在的环境变量。"""
    global _env_loaded
    if _env_loaded and path is None:
        return None
    env_path = path or _find_env_file()
    if env_path is None or not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
    _env_loaded = True
    return env_path


def get_config() -> Config:
    """从环境变量（含 .env）读取配置；任何缺失项使用默认值。"""
    load_env_file()
    return Config(
        openai_base_url=_env_str("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        openai_api_key=_env_str("OPENAI_API_KEY", "not-needed"),
        model_name=_env_str("MODEL_NAME", "openpangu-7b"),
        max_turns=_env_int("AGENT_MAX_TURNS", 6),
        temperature=_env_float("AGENT_TEMPERATURE", 0.1),
        review_mode=_env_str("REVIEW_MODE", "code"),
        request_timeout=_env_float("REQUEST_TIMEOUT", 60.0),
        debug=_env_bool("AGENT_DEBUG", False),
        disable_proxy=_env_bool("AGENT_DISABLE_PROXY", True),
        model_call_limit=_env_int("AGENT_MODEL_CALL_LIMIT", 8),
        tool_call_limit=_env_int("AGENT_TOOL_CALL_LIMIT", 10),
    )
