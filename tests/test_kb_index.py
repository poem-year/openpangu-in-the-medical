"""kb 索引层测试：建库、增量复用、删除、全量重建、缺失语义。"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from kb.build import build_index
from kb.config import KBConfig
from kb.corpus import CorpusError
from kb.embed import FakeBackend
from kb.index import CHUNKS_NAME, NumpyIndex, VECTORS_NAME


def _silent(*_args, **_kwargs) -> None:
    return None


def _entry(rel: str, doc_id: str) -> dict:
    return {
        "file": rel,
        "doc_id": doc_id,
        "name": f"{rel} 出处",
        "version": "V1",
        "category": "测试科",
    }


def _write_corpus(corpus, files: dict[str, str], entries: list[dict]) -> None:
    corpus.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        (corpus / rel).write_text(text, encoding="utf-8")
    (corpus / "sources.yaml").write_text(
        yaml.safe_dump({"files": entries}, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


@pytest.fixture()
def kb_paths(tmp_path):
    corpus = tmp_path / "corpus"
    index = tmp_path / "index"
    files = {
        "a.md": "# 高血压\n\n限盐每日不超过 5 克。\n",
        "b.md": "# 冠心病\n\n活动后胸骨后压榨样疼痛。\n",
    }
    _write_corpus(corpus, files, [_entry("a.md", "htn"), _entry("b.md", "chd")])
    return corpus, index, files


def _config(corpus, index) -> KBConfig:
    return KBConfig(corpus_dir=corpus, index_dir=index, embed_batch_size=8)


class TestBuild:
    def test_build_then_load_roundtrip(self, kb_paths):
        corpus, index, _ = kb_paths
        report, errors = build_index(_config(corpus, index), backend=FakeBackend(), log=_silent)
        assert not errors
        loaded = NumpyIndex.load(index)
        assert loaded is not None
        assert loaded.size == report["n_chunks"] == 2
        assert loaded.dim == 8
        assert (index / VECTORS_NAME).is_file()
        assert (index / CHUNKS_NAME).is_file()
        assert {record.doc_id for record in loaded.records} == {"htn", "chd"}

    def test_empty_registry_raises(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "sources.yaml").write_text("files: []\n", encoding="utf-8")
        with pytest.raises(CorpusError):
            build_index(_config(corpus, tmp_path / "index"), backend=FakeBackend(), log=_silent)

    def test_missing_registry_raises(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("# 标题\n\n正文。\n", encoding="utf-8")
        with pytest.raises(CorpusError):
            build_index(_config(corpus, tmp_path / "index"), backend=FakeBackend(), log=_silent)


class TestIncremental:
    def test_second_build_reuses_all_vectors(self, kb_paths):
        corpus, index, _ = kb_paths
        config = _config(corpus, index)
        build_index(config, backend=FakeBackend(), log=_silent)
        before = NumpyIndex.load(index).vectors

        report, _ = build_index(config, backend=FakeBackend(), log=_silent)
        after = NumpyIndex.load(index)
        assert report["embedded_vectors"] == 0
        assert report["reused_vectors"] == report["n_chunks"]
        assert report["file_status"] == {"a.md": "unchanged", "b.md": "unchanged"}
        assert np.allclose(before, after.vectors)

    def test_changed_file_is_reembedded_only(self, kb_paths):
        corpus, index, _ = kb_paths
        config = _config(corpus, index)
        build_index(config, backend=FakeBackend(), log=_silent)

        (corpus / "a.md").write_text("# 高血压\n\n限盐每日不超过 6 克。\n", encoding="utf-8")
        report, _ = build_index(config, backend=FakeBackend(), log=_silent)
        assert report["file_status"]["a.md"] == "changed"
        assert report["file_status"]["b.md"] == "unchanged"
        assert report["embedded_vectors"] == 1
        assert report["reused_vectors"] == 1

    def test_removed_file_drops_its_chunks(self, kb_paths):
        corpus, index, _ = kb_paths
        config = _config(corpus, index)
        build_index(config, backend=FakeBackend(), log=_silent)
        old_ids = {record.chunk_id for record in NumpyIndex.load(index).records}

        (corpus / "b.md").unlink()
        (corpus / "sources.yaml").write_text(
            yaml.safe_dump({"files": [_entry("a.md", "htn")]}, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        report, _ = build_index(config, backend=FakeBackend(), log=_silent)
        new_ids = {record.chunk_id for record in NumpyIndex.load(index).records}
        assert report["removed_chunks"] >= 1
        assert new_ids < old_ids
        assert report["file_status"]["b.md"] == "removed"

    def test_rebuild_matches_incremental(self, kb_paths):
        corpus, index, _ = kb_paths
        config = _config(corpus, index)
        build_index(config, backend=FakeBackend(), log=_silent)
        incremental_ids = [record.chunk_id for record in NumpyIndex.load(index).records]

        report, _ = build_index(config, backend=FakeBackend(), rebuild=True, log=_silent)
        assert report["mode"] == "rebuild"
        assert report["embedded_vectors"] == report["n_chunks"]
        assert report["reused_vectors"] == 0
        assert [record.chunk_id for record in NumpyIndex.load(index).records] == incremental_ids

    def test_chunk_ids_reproducible_across_rebuilds(self, kb_paths):
        corpus, index, _ = kb_paths
        config = _config(corpus, index)
        build_index(config, backend=FakeBackend(), rebuild=True, log=_silent)
        first = [record.chunk_id for record in NumpyIndex.load(index).records]
        build_index(config, backend=FakeBackend(), rebuild=True, log=_silent)
        second = [record.chunk_id for record in NumpyIndex.load(index).records]
        assert first == second

    def test_model_change_forces_full_reembed(self, kb_paths):
        corpus, index, _ = kb_paths
        build_index(_config(corpus, index), backend=FakeBackend(), log=_silent)
        changed = KBConfig(corpus_dir=corpus, index_dir=index, embed_model="other-model")
        report, _ = build_index(changed, backend=FakeBackend(), log=_silent)
        assert report["embedded_vectors"] == report["n_chunks"]


class TestMissing:
    def test_load_returns_none_when_absent(self, tmp_path):
        assert NumpyIndex.load(tmp_path / "没有这个目录") is None

    def test_load_returns_none_when_incomplete(self, tmp_path):
        (tmp_path / "index").mkdir()
        (tmp_path / "index" / VECTORS_NAME).write_bytes(b"")
        assert NumpyIndex.load(tmp_path / "index") is None
