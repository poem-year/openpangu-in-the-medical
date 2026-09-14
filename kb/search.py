"""检索入口：向量 + BM25 → RRF 融合 → 阈值过滤 → 同文档配额 → top-k。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kb.config import KBConfig, STRATEGY_HYBRID, STRATEGY_VECTOR, get_config
from kb.embed import BgeM3Backend, EmbeddingBackend
from kb.index import MANIFEST_NAME, NumpyIndex


class KbIndexMissingError(RuntimeError):
    """索引缺失/为空：属于服务不可用，工具层应降级为「检索暂不可用」而不是「查不到资料」。"""


@dataclass(frozen=True)
class RetrievedChunk:
    """检索结果。字段与《智能体接口规范.md》§2.2 的 Evidence 一一对应。"""

    chunk_id: str
    text: str
    source: str
    section: str
    score: float


_index_cache: dict[tuple[str, float, int], NumpyIndex] = {}
_backend_cache: dict[tuple, EmbeddingBackend] = {}


def _index_signature(index_dir: Path) -> float:
    """用 manifest 的 mtime 当缓存键：重建索引后自动重新加载。"""
    manifest = index_dir / MANIFEST_NAME
    try:
        return manifest.stat().st_mtime
    except OSError:
        return 0.0


def clear_cache() -> None:
    """清空索引与嵌入模型缓存（重建索引、测试隔离时用）。"""
    _index_cache.clear()
    _backend_cache.clear()


def load_index(config: KBConfig | None = None) -> NumpyIndex:
    """加载索引；缺失或为空时抛 KbIndexMissingError。"""
    cfg = config or get_config()
    index_dir = Path(cfg.index_dir)
    signature = (str(index_dir), _index_signature(index_dir), 0)
    cached = _index_cache.get(signature)
    if cached is not None:
        return cached
    index = NumpyIndex.load(index_dir)
    if index is None:
        raise KbIndexMissingError(
            f"未找到可用索引：{index_dir}\n"
            f"请先建库：.venv/bin/python -m kb.build（语料放在 {cfg.corpus_dir}）"
        )
    if index.size == 0:
        raise KbIndexMissingError(f"索引为空：{index_dir}（语料目录里没有可用内容）")
    _index_cache[signature] = index
    return index


def get_backend(
    config: KBConfig | None = None, backend: EmbeddingBackend | None = None
) -> EmbeddingBackend:
    """嵌入后端（惰性加载、进程内复用）。"""
    if backend is not None:
        return backend
    cfg = config or get_config()
    key = (
        cfg.embed_model,
        cfg.device,
        cfg.embed_batch_size,
        cfg.query_prompt,
        cfg.doc_prompt,
    )
    if key not in _backend_cache:
        _backend_cache[key] = BgeM3Backend(
            cfg.embed_model,
            device=cfg.device,
            batch_size=cfg.embed_batch_size,
            query_prompt=cfg.query_prompt,
            doc_prompt=cfg.doc_prompt,
        )
    return _backend_cache[key]


def _rrf(rank_lists: list[list[int]], rrf_k: int) -> dict[int, float]:
    """Reciprocal Rank Fusion：多路结果按名次融合，分数只用于排序。"""
    fused: dict[int, float] = {}
    for rows in rank_lists:
        for rank, row in enumerate(rows, start=1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (rrf_k + rank)
    return fused


def _rank(
    query: str,
    *,
    k: int,
    strategy: str,
    threshold: float | None,
    cfg: KBConfig,
    backend: EmbeddingBackend | None = None,
) -> list[dict[str, Any]]:
    """核心排序逻辑。返回带调试字段的命中列表（已按最终顺序）。"""
    index = load_index(cfg)
    embedder = get_backend(cfg, backend)
    pool = min(max(cfg.candidate_pool, 5 * k), index.size)

    query_vector = embedder.encode_query(query)
    cosines = index.cosines(query_vector)
    vector_top = [row for row, _ in index.search(query_vector, pool)]

    if strategy == STRATEGY_VECTOR:
        ranked_rows = sorted(vector_top, key=lambda row: -cosines[row])
        fused = {row: float(cosines[row]) for row in ranked_rows}
        bm25_rank: dict[int, int] = {}
    else:
        bm25_top = [row for row, _ in index.bm25.top(query, pool)]
        fused = _rrf([vector_top, bm25_top], cfg.rrf_k)
        ranked_rows = sorted(fused, key=lambda row: (-fused[row], -cosines[row]))
        bm25_rank = {row: rank for rank, row in enumerate(bm25_top, start=1)}

    vector_rank = {row: rank for rank, row in enumerate(vector_top, start=1)}

    hits: list[dict[str, Any]] = []
    per_doc: dict[str, int] = {}
    for row in ranked_rows:
        cosine = float(cosines[row])
        if threshold is not None and cosine < threshold:
            continue
        record = index.records[row]
        if per_doc.get(record.doc_id, 0) >= cfg.doc_quota:
            continue
        per_doc[record.doc_id] = per_doc.get(record.doc_id, 0) + 1
        hits.append(
            {
                "chunk_id": record.chunk_id,
                "text": record.text,
                "source": record.source_name,
                "section": record.section,
                "doc_id": record.doc_id,
                "score": round(float(fused[row]), 6),
                "cosine": round(cosine, 6),
                "vector_rank": vector_rank.get(row),
                "bm25_rank": bm25_rank.get(row),
            }
        )
        if len(hits) >= k:
            break
    return hits


def retrieve(
    query: str,
    k: int | None = None,
    *,
    strategy: str | None = None,
    threshold: float | None = None,
    config: KBConfig | None = None,
    backend: EmbeddingBackend | None = None,
) -> list[RetrievedChunk]:
    """检索知识库。

    - strategy：`hybrid`（向量 + BM25 融合，默认）或 `vector`（纯向量，L2 基线用）；
    - threshold：向量余弦下限；不传时 hybrid 用配置值、vector 不过滤；
    - 无结果返回 `[]`；索引缺失抛 KbIndexMissingError；query 为空抛 ValueError。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")
    hits = retrieve_debug(
        query, k=k, strategy=strategy, threshold=threshold, config=config, backend=backend
    )
    return [
        RetrievedChunk(
            chunk_id=hit["chunk_id"],
            text=hit["text"],
            source=hit["source"],
            section=hit["section"],
            score=float(hit["score"]),
        )
        for hit in hits
    ]


def retrieve_debug(
    query: str,
    k: int | None = None,
    *,
    strategy: str | None = None,
    threshold: float | None = None,
    config: KBConfig | None = None,
    backend: EmbeddingBackend | None = None,
) -> list[dict[str, Any]]:
    """检索并保留调试字段（向量余弦、各路名次、融合分），供 kb.inspect 使用。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空")
    cfg = (config or get_config()).validated()
    use_strategy = strategy or cfg.strategy
    if use_strategy not in (STRATEGY_VECTOR, STRATEGY_HYBRID):
        raise ValueError(f"strategy 必须是 vector 或 hybrid，收到：{use_strategy!r}")
    use_k = int(k) if k else cfg.top_k
    if use_k < 1:
        raise ValueError(f"k 必须 ≥1，收到：{k}")

    if threshold is None and use_strategy == STRATEGY_HYBRID:
        use_threshold: float | None = cfg.score_threshold
    else:
        use_threshold = threshold

    return _rank(
        query,
        k=use_k,
        strategy=use_strategy,
        threshold=use_threshold,
        cfg=cfg,
        backend=backend,
    )
