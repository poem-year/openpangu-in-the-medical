"""kb 检索层测试：排序、RRF 融合、阈值、同文档配额、空结果与缺失语义。"""

from __future__ import annotations

import numpy as np
import pytest

from kb.config import KBConfig
from kb.embed import FakeBackend, l2_normalize
from kb.index import IndexRecord, write_index_atomically
from kb.search import KbIndexMissingError, clear_cache, retrieve, retrieve_debug


def _write_index(index_dir, rows: list[dict]) -> None:
    records = [
        IndexRecord(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            section=row.get("section", "章节"),
            seq=index,
            source_name=row.get("source", "测试出处"),
            source_version="V1",
            category="测试科",
            file_sha1="deadbeef",
            rel_path="t.md",
        )
        for index, row in enumerate(rows)
    ]
    vectors = (
        l2_normalize(np.asarray([row["vector"] for row in rows], dtype=np.float32))
        if rows
        else np.zeros((0, 1), dtype=np.float32)
    )
    write_index_atomically(
        index_dir,
        vectors,
        records,
        {"embed_model": "fake", "dim": int(vectors.shape[1]), "n_chunks": len(records)},
        {},
    )


def _config(index_dir, **kwargs) -> KBConfig:
    base = {"top_k": 5, "candidate_pool": 10, "doc_quota": 2, "score_threshold": 0.0}
    base.update(kwargs)
    return KBConfig(index_dir=index_dir, **base).validated()


@pytest.fixture(autouse=True)
def _clear_index_cache():
    clear_cache()
    yield
    clear_cache()


class TestVectorRanking:
    def test_orders_by_cosine_and_reports_descending(self, tmp_path):
        rows = [
            {"chunk_id": "kb::a::1", "doc_id": "a", "text": "高相关", "vector": [1.0, 0.0]},
            {"chunk_id": "kb::b::2", "doc_id": "b", "text": "中相关", "vector": [0.5, 0.866]},
            {"chunk_id": "kb::c::3", "doc_id": "c", "text": "低相关", "vector": [0.0, 1.0]},
        ]
        _write_index(tmp_path, rows)
        backend = FakeBackend(vectors={"查询": [1.0, 0.0]})
        hits = retrieve("查询", k=3, strategy="vector", config=_config(tmp_path), backend=backend)
        assert [hit.chunk_id for hit in hits] == ["kb::a::1", "kb::b::2", "kb::c::3"]
        assert [round(hit.score, 3) for hit in hits] == [1.0, 0.5, 0.0]
        assert hits[0].source == "测试出处" and hits[0].section == "章节"

    def test_k_is_an_upper_bound(self, tmp_path):
        rows = [
            {"chunk_id": f"kb::{doc}::{i}", "doc_id": doc, "text": f"正文{i}", "vector": [1.0, 0.0]}
            for i, doc in enumerate("abc")
        ]
        _write_index(tmp_path, rows)
        backend = FakeBackend(vectors={"查询": [1.0, 0.0]})
        hits = retrieve("查询", k=1, strategy="vector", config=_config(tmp_path), backend=backend)
        assert len(hits) == 1


class TestThreshold:
    def test_filters_low_similarity(self, tmp_path):
        rows = [
            {"chunk_id": "kb::a::1", "doc_id": "a", "text": "高相关", "vector": [1.0, 0.0]},
            {"chunk_id": "kb::b::2", "doc_id": "b", "text": "弱相关", "vector": [0.3, 0.954]},
        ]
        _write_index(tmp_path, rows)
        backend = FakeBackend(vectors={"查询": [1.0, 0.0]})
        hits = retrieve(
            "查询", k=2, config=_config(tmp_path, score_threshold=0.5), backend=backend
        )
        assert [hit.chunk_id for hit in hits] == ["kb::a::1"]

    def test_no_candidate_returns_empty_list(self, tmp_path):
        rows = [{"chunk_id": "kb::a::1", "doc_id": "a", "text": "无关", "vector": [0.0, 1.0]}]
        _write_index(tmp_path, rows)
        backend = FakeBackend(vectors={"查询": [1.0, 0.0]})
        hits = retrieve(
            "查询", k=2, config=_config(tmp_path, score_threshold=0.5), backend=backend
        )
        assert hits == []

    def test_vector_strategy_does_not_filter_by_default(self, tmp_path):
        rows = [{"chunk_id": "kb::a::1", "doc_id": "a", "text": "无关", "vector": [0.0, 1.0]}]
        _write_index(tmp_path, rows)
        backend = FakeBackend(vectors={"查询": [1.0, 0.0]})
        hits = retrieve("查询", k=2, strategy="vector", config=_config(tmp_path), backend=backend)
        assert len(hits) == 1


class TestHybrid:
    def _corpus(self):
        rows = [
            {
                "chunk_id": "kb::vecA::1",
                "doc_id": "vecA",
                "text": "语义最接近的甲",
                "vector": [0.5, 0.866],
            },
            {
                "chunk_id": "kb::vecB::2",
                "doc_id": "vecB",
                "text": "语义次接近的乙",
                "vector": [0.4, 0.9165],
            },
            {
                "chunk_id": "kb::lex::1",
                "doc_id": "lex",
                "text": "含罕见词 XYZ 的资料",
                "vector": [0.3, 0.954],
            },
        ]
        rows += [
            {
                "chunk_id": f"kb::fill{i}::{i}",
                "doc_id": f"fill{i}",
                "text": f"填充内容 {i}",
                "vector": [0.0, 1.0],
            }
            for i in range(27)
        ]
        return rows

    def test_bm25_lifts_lexical_hit_above_higher_cosine(self, tmp_path):
        _write_index(tmp_path, self._corpus())
        backend = FakeBackend(vectors={"罕见词 XYZ": [1.0, 0.0]})
        config = _config(tmp_path, top_k=3, candidate_pool=3)

        vector_ids = [
            hit.chunk_id
            for hit in retrieve("罕见词 XYZ", k=3, strategy="vector", config=config, backend=backend)
        ]
        assert vector_ids == ["kb::vecA::1", "kb::vecB::2", "kb::lex::1"], "纯向量按语义排序"

        hybrid_ids = [
            hit.chunk_id for hit in retrieve("罕见词 XYZ", k=3, config=config, backend=backend)
        ]
        assert hybrid_ids[0] == "kb::lex::1", "两路都命中的应排到最前"

    def test_debug_exposes_both_ranks(self, tmp_path):
        _write_index(tmp_path, self._corpus())
        backend = FakeBackend(vectors={"罕见词 XYZ": [1.0, 0.0]})
        hits = retrieve_debug(
            "罕见词 XYZ", k=1, config=_config(tmp_path, top_k=1, candidate_pool=3), backend=backend
        )
        assert hits[0]["bm25_rank"] == 1
        assert hits[0]["vector_rank"] == 3
        assert hits[0]["cosine"] == pytest.approx(0.3, abs=1e-3)


class TestDocQuota:
    def test_limits_chunks_per_document(self, tmp_path):
        rows = [
            {"chunk_id": "kb::a::1", "doc_id": "a", "text": "甲一", "vector": [1.0, 0.0]},
            {"chunk_id": "kb::a::2", "doc_id": "a", "text": "甲二", "vector": [0.99, 0.141]},
            {"chunk_id": "kb::a::3", "doc_id": "a", "text": "甲三", "vector": [0.98, 0.199]},
            {"chunk_id": "kb::b::4", "doc_id": "b", "text": "乙一", "vector": [0.9, 0.436]},
        ]
        _write_index(tmp_path, rows)
        backend = FakeBackend(vectors={"查询": [1.0, 0.0]})
        hits = retrieve(
            "查询", k=4, strategy="vector", config=_config(tmp_path, doc_quota=2), backend=backend
        )
        ids = [hit.chunk_id for hit in hits]
        assert ids == ["kb::a::1", "kb::a::2", "kb::b::4"]
        assert "kb::a::3" not in ids


class TestErrors:
    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(KbIndexMissingError):
            retrieve("查询", config=_config(tmp_path / "不存在"), backend=FakeBackend())

    def test_empty_query_raises(self, tmp_path):
        rows = [{"chunk_id": "kb::a::1", "doc_id": "a", "text": "x", "vector": [1.0]}]
        _write_index(tmp_path, rows)
        with pytest.raises(ValueError):
            retrieve("   ", config=_config(tmp_path), backend=FakeBackend())

    def test_bad_strategy_raises(self, tmp_path):
        rows = [{"chunk_id": "kb::a::1", "doc_id": "a", "text": "x", "vector": [1.0]}]
        _write_index(tmp_path, rows)
        with pytest.raises(ValueError):
            retrieve("查询", strategy="暴力", config=_config(tmp_path), backend=FakeBackend())
