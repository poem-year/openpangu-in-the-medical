"""kb 命令行的可测部分：阈值标定、预检索、调试输出。

这几条以前只有手工跑过，回归时容易悄悄坏掉。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from kb.config import KBConfig
from kb.embed import FakeBackend, l2_normalize
from kb.index import IndexRecord, write_index_atomically
from kb.search import clear_cache


def _write_index(index_dir: Path, rows: list[dict]) -> None:
    records = [
        IndexRecord(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            section="章节",
            seq=index,
            source_name="测试出处",
            source_version="V1",
            category="科",
            file_sha1="x",
            rel_path="t.md",
        )
        for index, row in enumerate(rows)
    ]
    vectors = l2_normalize(np.asarray([row["vector"] for row in rows], dtype=np.float32))
    write_index_atomically(
        index_dir, vectors, records,
        {"embed_model": "fake", "dim": int(vectors.shape[1]), "n_chunks": len(records)}, {},
    )


@pytest.fixture()
def toy_index(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    _write_index(
        index_dir,
        [
            {"chunk_id": "kb::a::1", "doc_id": "a", "text": "高血压限盐", "vector": [1.0, 0.0]},
            {"chunk_id": "kb::b::2", "doc_id": "b", "text": "冠心病胸痛", "vector": [0.3, 0.954]},
        ],
    )
    clear_cache()
    yield index_dir
    clear_cache()


class TestCalibrate:
    def _stub(self, monkeypatch, threshold_to_hits):
        """用桩替换检索：按传入阈值决定返回几条，并记录收到的阈值。"""
        import kb.calibrate as cal

        seen: list[float | None] = []

        def fake(question, k=None, *, strategy=None, threshold=None, config=None, backend=None):
            seen.append(threshold)
            count = threshold_to_hits(threshold)
            return [
                {"chunk_id": f"kb::a::{i}", "doc_id": "a", "text": "x", "source": "s",
                 "section": "sec", "score": 1.0, "cosine": 0.9}
                for i in range(count)
            ]

        monkeypatch.setattr(cal, "retrieve_debug", fake)
        return cal, seen

    def test_baseline_row_really_skips_threshold(self, monkeypatch):
        """回归：`threshold=None` 这一行必须是「不过滤」，不能用配置里的阈值。

        以前直接把 None 传下去，而 kb.search 里 None 的语义是「用配置阈值」，
        于是标定表第一行被误标成基线。
        """
        cal, seen = self._stub(monkeypatch, lambda t: 2 if (t is not None and t < 0) else 0)
        items = [
            {"id": "Q1", "question": "高血压", "type": "relevant", "expected_doc_ids": ["a"]},
            {"id": "N1", "question": "今天天气", "type": "irrelevant"},
        ]
        row = cal.evaluate(items, k=5, threshold=None)
        assert seen and all(value is not None and value < 0 for value in seen), "不过滤必须传一个真正不筛的阈值"
        assert row["recall_at_k"] == 1.0 and row["avg_returned"] == 2.0
        assert row["false_hit_rate"] == 1.0, "不过滤时无关问题也会被召回"

    def test_explicit_threshold_is_passed_through(self, monkeypatch):
        cal, seen = self._stub(monkeypatch, lambda t: 1 if t is not None and t <= 0.5 else 0)
        items = [{"id": "Q1", "question": "高血压", "type": "relevant", "expected_doc_ids": ["a"]}]
        row = cal.evaluate(items, k=5, threshold=0.5)
        assert seen == [0.5]
        assert row["recall_at_k"] == 1.0 and row["avg_returned"] == 1.0

    def test_chunk_level_gold_takes_priority(self, monkeypatch):
        """给了 expected_chunk_ids 就按 chunk 判，不能被粗粒度的 doc 级 gold 蒙混过去。"""
        import kb.calibrate as cal

        def fake(question, k=None, *, strategy=None, threshold=None, config=None, backend=None):
            return [
                {"chunk_id": "kb::a::999", "doc_id": "a", "section": "别的病",
                 "text": "x", "source": "s", "score": 1.0, "cosine": 0.9}
            ]

        monkeypatch.setattr(cal, "retrieve_debug", fake)
        # 命中了同一个 doc_id，但不是期望的那一条 → chunk 级 gold 应判为未召回
        chunk_level = cal.evaluate(
            [{"id": "Q1", "question": "高血压", "type": "relevant",
              "expected_chunk_ids": ["kb::a::1"], "expected_doc_ids": ["a"]}],
            k=5, threshold=None,
        )
        assert chunk_level["recall_at_k"] == 0.0

        doc_level = cal.evaluate(
            [{"id": "Q2", "question": "高血压", "type": "relevant", "expected_doc_ids": ["a"]}],
            k=5, threshold=None,
        )
        assert doc_level["recall_at_k"] == 1.0


class TestEvalRetrieve:
    def test_writes_contexts_file(self, toy_index, tmp_path, monkeypatch):
        import kb.eval_retrieve as er

        monkeypatch.setattr(er, "get_config", lambda: KBConfig(index_dir=toy_index))
        monkeypatch.setattr("kb.search.get_backend", lambda *a, **k: FakeBackend(vectors={"高血压": [1.0, 0.0]}))
        clear_cache()
        items_path = tmp_path / "items.jsonl"
        items_path.write_text(
            json.dumps({"id": "T1", "question": "高血压"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "contexts.jsonl"
        monkeypatch.setattr(
            sys, "argv",
            ["kb.eval_retrieve", "--input", str(items_path), "--out", str(out_path), "--layer", "L2"],
        )
        assert er.main() == 0
        row = json.loads(out_path.read_text(encoding="utf-8").strip())
        assert row["id"] == "T1" and row["strategy"] == "vector"
        # L2 = 纯向量、不过滤，所以两条都会返回（这正是 L2/L3 消融要的基线行为）
        assert row["retrieved_chunk_ids"] == ["kb::a::1", "kb::b::2"]

    def test_missing_index_returns_error_code(self, tmp_path, monkeypatch):
        import kb.eval_retrieve as er

        monkeypatch.setattr(er, "get_config", lambda: KBConfig(index_dir=tmp_path / "无索引"))
        clear_cache()
        items_path = tmp_path / "items.jsonl"
        items_path.write_text('{"id": "T1", "question": "高血压"}\n', encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv",
            ["kb.eval_retrieve", "--input", str(items_path), "--out", str(tmp_path / "c.jsonl")],
        )
        assert er.main() == 2, "索引缺失应返回 2，让评测脚本快速失败"
