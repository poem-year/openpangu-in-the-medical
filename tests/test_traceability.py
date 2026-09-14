"""引用可追溯率的解析：chunk_id 必须能被抠出来，否则指标永远是 0。

这条指标踩过一次真坑：chunk_id 形如 `kb::baike-38::3badd7513c40`，
而旧正则的字符集不含冒号，模型就算按提示词写了 `[chunk_id: kb::…]` 也匹配不到。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from traceability import extract_citations, locatable_rate, traceability_rate  # noqa: E402


class TestExtractCitations:
    def test_reads_chunk_id_written_as_instructed(self):
        answer = "考虑嵌顿疝。[chunk_id: kb::baike-38::3badd7513c40]"
        assert extract_citations(answer) == ["kb::baike-38::3badd7513c40"]

    def test_reads_bare_chunk_id(self):
        answer = "依据 kb::baike-44::77461f0a3370 这段资料。"
        assert extract_citations(answer) == ["kb::baike-44::77461f0a3370"]

    def test_reads_multiple_and_dedupes(self):
        answer = (
            "[chunk_id: kb::a::111111111111] 与 [chunk_id: kb::b::222222222222]；"
            "再引一次 [chunk_id: kb::a::111111111111]"
        )
        assert extract_citations(answer) == ["kb::a::111111111111", "kb::b::222222222222"]

    def test_trailing_punctuation_is_stripped(self):
        assert extract_citations("见 [chunk_id: kb::a::111111111111]。") == ["kb::a::111111111111"]

    def test_numbered_marker_maps_to_evidence(self):
        """提示词把资料编号成 1..k，模型用 [1] 指代第 1 条证据。"""
        ids = ["kb::a::111111111111", "kb::b::222222222222"]
        assert extract_citations("根据参考资料[1]，考虑…", evidence_ids=ids) == [ids[0]]

    def test_out_of_range_marker_is_ignored(self):
        ids = ["kb::a::111111111111"]
        assert extract_citations("见[9]", evidence_ids=ids) == []

    def test_no_citation_returns_empty(self):
        assert extract_citations("普通回答，没有引用。") == []


class TestLocatableRate:
    def test_all_locatable(self):
        ids = ["kb::a::1", "kb::b::2"]
        assert locatable_rate(["kb::a::1"], ids) == 1.0

    def test_half_locatable(self):
        assert locatable_rate(["kb::a::1", "kb::fake::9"], ["kb::a::1"]) == 0.5

    def test_none_when_no_citation(self):
        assert locatable_rate([], ["kb::a::1"]) is None


class TestTraceabilityRate:
    def test_counts_only_items_with_evidence(self):
        result = traceability_rate(
            [
                {"id": "1", "answer": "见 [chunk_id: kb::a::1]", "retrieved_chunk_ids": ["kb::a::1"]},
                {"id": "2", "answer": "无证据题", "retrieved_chunk_ids": []},
            ]
        )
        assert result["n"] == 1
        assert result["locatable_rate"] == 1.0

    def test_fabricated_citation_costs_the_item(self):
        result = traceability_rate(
            [{"id": "1", "answer": "见 [chunk_id: kb::fake::9]", "retrieved_chunk_ids": ["kb::a::1"]}]
        )
        assert result["locatable_rate"] == 0.0

    def test_unanswered_item_is_not_counted_as_traceable(self):
        result = traceability_rate(
            [{"id": "1", "answer": "没有引用任何资料", "retrieved_chunk_ids": ["kb::a::1"]}]
        )
        assert result["n"] == 1
        assert result["locatable_rate"] == 0.0
        assert result["traceability_rate"] == 0.0
