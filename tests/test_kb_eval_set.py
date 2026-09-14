"""kb.eval_set 的取样逻辑：跨文档、可复现、能等距截断。

取样错了不会报错，只会让标定出来的阈值悄悄偏掉，所以单独回归。
"""

from __future__ import annotations

from dataclasses import dataclass

from kb.eval_set import sample_sections


@dataclass
class _Record:
    doc_id: str
    section: str
    chunk_id: str
    text: str = "正文"


class _Index:
    def __init__(self, records: list[_Record]) -> None:
        self.records = records


def _index(n_docs: int = 10, sections_per_doc: int = 5) -> _Index:
    records: list[_Record] = []
    for doc in range(n_docs):
        for section in range(sections_per_doc):
            records.append(_Record(f"doc-{doc}", f"章节{section}", f"kb::doc-{doc}::{section}"))
    return _Index(records)


def _ids(samples: list[dict]) -> list[str]:
    return [sample["record"].chunk_id for sample in samples]


class TestSampleSections:
    def test_default_returns_every_section_once(self):
        samples = sample_sections(_index(), 1)
        assert len(samples) == 50

    def test_sampling_spreads_across_documents(self):
        """默认按索引顺序取会全落在第一个文档上，这是取样必须跨文档的原因。"""
        samples = sample_sections(_index(), 1, limit=10, max_per_doc=2)
        assert len({sample["doc_id"] for sample in samples}) >= 5

    def test_max_per_doc_is_respected(self):
        samples = sample_sections(_index(), 1, max_per_doc=1)
        docs = [sample["doc_id"] for sample in samples]
        assert len(docs) == len(set(docs))

    def test_limit_truncates(self):
        assert len(sample_sections(_index(), 1, limit=7)) == 7

    def test_same_seed_is_reproducible(self):
        first = sample_sections(_index(), 1, limit=10, max_per_doc=2, seed=1)
        second = sample_sections(_index(), 1, limit=10, max_per_doc=2, seed=1)
        assert _ids(first) == _ids(second)

    def test_different_seed_changes_pick(self):
        first = sample_sections(_index(), 1, limit=10, max_per_doc=2, seed=1)
        second = sample_sections(_index(), 1, limit=10, max_per_doc=2, seed=2)
        assert _ids(first) != _ids(second)

    def test_samples_carry_their_own_chunk_id(self):
        """gold 必须来自素材本身：问题要能指回它出自哪一条切块。"""
        index = _Index(
            [_Record("d1", "章节A", "kb::d1::aaa"), _Record("d2", "章节B", "kb::d2::bbb")]
        )
        samples = sample_sections(index, 1)
        assert {s["record"].chunk_id for s in samples} == {"kb::d1::aaa", "kb::d2::bbb"}
