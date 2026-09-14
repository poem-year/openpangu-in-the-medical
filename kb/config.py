"""kb 包配置：环境变量 + 默认值（写法对齐 agent/config.py）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = REPO_ROOT / "kb" / "corpus"
DEFAULT_INDEX_DIR = REPO_ROOT / "kb" / "index"

STRATEGY_VECTOR = "vector"
STRATEGY_HYBRID = "hybrid"
VALID_STRATEGIES = (STRATEGY_VECTOR, STRATEGY_HYBRID)


@dataclass
class KBConfig:
    """检索配置。默认值见《环境说明.md》与 `kb/README.md`。"""

    corpus_dir: Path = DEFAULT_CORPUS_DIR
    index_dir: Path = DEFAULT_INDEX_DIR
    embed_model: str = "BAAI/bge-m3"
    device: str = "cpu"
    top_k: int = 5
    # 由 kb.calibrate 在 201 病种测试语料上标定：0.50~0.60 是平台区（Recall@5 100%、
    # 无关误召 0%），0.65 起开始掉召回。取平台中点，语料换新后必须重新标定。
    score_threshold: float = 0.55
    strategy: str = STRATEGY_HYBRID
    rrf_k: int = 60
    doc_quota: int = 2
    candidate_pool: int = 50
    embed_batch_size: int = 32
    chunk_target_max: int = 500
    chunk_split_size: int = 600
    chunk_overlap: int = 80
    query_prompt: str = ""
    doc_prompt: str = ""

    def validated(self) -> "KBConfig":
        """校验取值；非法值直接报错，不静默回退。"""
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(f"KB_STRATEGY 必须是 {VALID_STRATEGIES} 之一，收到：{self.strategy!r}")
        if self.top_k < 1:
            raise ValueError(f"KB_TOP_K 必须 ≥1，收到：{self.top_k}")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(f"KB_SCORE_THRESHOLD 必须落在 0~1，收到：{self.score_threshold}")
        if self.rrf_k < 1:
            raise ValueError(f"KB_RRF_K 必须 ≥1，收到：{self.rrf_k}")
        if self.doc_quota < 1:
            raise ValueError(f"KB_DOC_QUOTA 必须 ≥1，收到：{self.doc_quota}")
        if self.candidate_pool < self.top_k:
            raise ValueError(
                f"KB_CANDIDATE_POOL（{self.candidate_pool}）不能小于 KB_TOP_K（{self.top_k}）"
            )
        return self


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


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def get_config() -> KBConfig:
    """从环境变量读取配置；缺失项用默认值，非法值抛 ValueError。"""
    cfg = KBConfig(
        corpus_dir=_env_path("KB_CORPUS_DIR", DEFAULT_CORPUS_DIR),
        index_dir=_env_path("KB_INDEX_DIR", DEFAULT_INDEX_DIR),
        embed_model=_env_str("KB_EMBED_MODEL", "BAAI/bge-m3"),
        device=_env_str("KB_DEVICE", "cpu"),
        top_k=_env_int("KB_TOP_K", 5),
        score_threshold=_env_float("KB_SCORE_THRESHOLD", 0.55),
        strategy=_env_str("KB_STRATEGY", STRATEGY_HYBRID),
        rrf_k=_env_int("KB_RRF_K", 60),
        doc_quota=_env_int("KB_DOC_QUOTA", 2),
        candidate_pool=_env_int("KB_CANDIDATE_POOL", 50),
        embed_batch_size=_env_int("KB_EMBED_BATCH_SIZE", 32),
        chunk_target_max=_env_int("KB_CHUNK_TARGET_MAX", 500),
        chunk_split_size=_env_int("KB_CHUNK_SPLIT_SIZE", 600),
        chunk_overlap=_env_int("KB_CHUNK_OVERLAP", 80),
        query_prompt=_env_str("KB_QUERY_PROMPT", ""),
        doc_prompt=_env_str("KB_DOC_PROMPT", ""),
    )
    return cfg.validated()
