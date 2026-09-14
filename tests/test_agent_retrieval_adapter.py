"""agent/retrieval.py 适配层测试：kb 结果 → Evidence、降级语义、chunk_id 上报。"""

from __future__ import annotations

import pytest

import agent.retrieval as retrieval_module
from agent.recorder import TurnRecorder, reset_recorder, set_recorder
from agent.retrieval import retrieve, retrieve_evidence
from agent.schemas import Evidence
from kb.search import RetrievedChunk, clear_cache


def _chunk(chunk_id: str = "kb::cardio::abc123def456") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="限盐每日不超过 5 克。",
        source="中国高血压防治指南",
        section="心血管内科 / 高血压",
        score=0.83,
    )


class TestAdapter:
    def test_maps_all_fields_to_evidence(self, monkeypatch):
        monkeypatch.setattr("kb.search.retrieve", lambda query, k=None, config=None: [_chunk()])
        result = retrieve("高血压吃盐", 3)
        assert len(result) == 1
        item = result[0]
        assert isinstance(item, Evidence)
        assert item.chunk_id == "kb::cardio::abc123def456"
        assert item.text == "限盐每日不超过 5 克。"
        assert item.source == "中国高血压防治指南"
        assert item.section == "心血管内科 / 高血压"
        assert item.score == pytest.approx(0.83)

    def test_default_k_comes_from_config(self, monkeypatch):
        seen: dict = {}

        def fake(query, k=None, config=None):
            seen["k"] = k
            return []

        monkeypatch.setattr("kb.search.retrieve", fake)
        retrieve("高血压", k=None)
        assert seen["k"] == 5

    def test_illegal_k_falls_back_to_default(self, monkeypatch):
        seen: dict = {}

        def fake(query, k=None, config=None):
            seen["k"] = k
            return []

        monkeypatch.setattr("kb.search.retrieve", fake)
        retrieve("高血压", k=0)
        assert seen["k"] == 5

    def test_empty_query_raises_value_error(self):
        with pytest.raises(ValueError):
            retrieve("   ")

    def test_missing_index_raises(self, tmp_path, monkeypatch):
        from kb.search import KbIndexMissingError

        monkeypatch.setenv("KB_INDEX_DIR", str(tmp_path / "不存在"))
        clear_cache()
        with pytest.raises(KbIndexMissingError):
            retrieve("高血压")


class TestToolLayer:
    def test_empty_result_is_reported_as_not_found(self, monkeypatch):
        monkeypatch.setattr(retrieval_module, "retrieve", lambda query, k=5: [])
        message = retrieve_evidence("咳嗽持续的原因", 3)
        assert "未找到" in message
        assert "不要编造" in message

    def test_missing_index_degrades_to_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KB_INDEX_DIR", str(tmp_path / "不存在"))
        clear_cache()
        message = retrieve_evidence("咳嗽", 3)
        assert "不可用" in message
        assert "KbIndexMissingError" in message

    def test_success_records_real_chunk_ids(self, monkeypatch):
        monkeypatch.setattr(
            retrieval_module,
            "retrieve",
            lambda query, k=5: [_chunk("kb::x::1"), _chunk("kb::x::2")],
        )
        recorder = TurnRecorder()
        token = set_recorder(recorder)
        try:
            message = retrieve_evidence("高血压", 5)
        finally:
            reset_recorder(token)
        assert "kb::x::1" in message and "kb::x::2" in message
        assert recorder.retrieved_chunk_ids == frozenset({"kb::x::1", "kb::x::2"})
        assert recorder.used_tools == ["retrieve_evidence"]
