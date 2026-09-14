"""向量索引：接口 + numpy 暴力精确检索实现 + 落盘。

索引产物（全部在 `KB_INDEX_DIR` 下）：
    vectors.npy       float32 矩阵，行已 L2 归一化（点积即余弦）
    chunks.jsonl      每行一条 IndexRecord，行号与 vectors.npy 对应
    bm25.json         分词器版本与文档数（token 由正文重建，避免翻倍占盘）
    manifest.json     语料文件 sha1 指纹 + 建库配置，供增量判断
    build_report.json 每次建库的结果明细

换引擎（faiss / Chroma 等）只需另写一个实现 `VectorIndex` 协议的后端。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from kb.bm25 import TOKENIZER_VERSION, Bm25Index

VECTORS_NAME = "vectors.npy"
CHUNKS_NAME = "chunks.jsonl"
BM25_NAME = "bm25.json"
MANIFEST_NAME = "manifest.json"
REPORT_NAME = "build_report.json"
REQUIRED_FILES = (VECTORS_NAME, CHUNKS_NAME, MANIFEST_NAME)


@dataclass(frozen=True)
class IndexRecord:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    seq: int
    source_name: str
    source_version: str
    category: str
    file_sha1: str
    rel_path: str
    # 来源分级与对外发布过滤用字段（只增不改，缺省为空）
    license: str = ""
    language: str = ""
    tier: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "IndexRecord":
        return cls(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload.get("doc_id", "")),
            text=str(payload.get("text", "")),
            section=str(payload.get("section", "")),
            seq=int(payload.get("seq", 0)),
            source_name=str(payload.get("source_name", "")),
            source_version=str(payload.get("source_version", "")),
            category=str(payload.get("category", "")),
            file_sha1=str(payload.get("file_sha1", "")),
            rel_path=str(payload.get("rel_path", "")),
            license=str(payload.get("license", "")),
            language=str(payload.get("language", "")),
            tier=str(payload.get("tier", "")),
        )


class VectorIndex(Protocol):
    """向量索引协议（预留换引擎的接口）。"""

    def cosines(self, query_vector: np.ndarray) -> np.ndarray: ...

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]: ...


class NumpyIndex:
    """暴力精确检索：万级 chunk 下内存几十 MB、单次约 10ms，召回是精确的。"""

    def __init__(
        self,
        vectors: np.ndarray,
        records: list[IndexRecord],
        manifest: dict | None = None,
    ) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.records = records
        self.manifest = manifest or {}
        if self.vectors.size and self.vectors.shape[0] != len(records):
            raise ValueError(
                f"vectors 行数（{self.vectors.shape[0]}）与记录数（{len(records)}）不一致"
            )
        self._bm25: Bm25Index | None = None

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    @property
    def bm25(self) -> Bm25Index:
        if self._bm25 is None:
            self._bm25 = Bm25Index([record.text for record in self.records])
        return self._bm25

    def cosines(self, query_vector: np.ndarray) -> np.ndarray:
        if not self.size:
            return np.zeros(0, dtype=np.float32)
        return self.vectors @ np.asarray(query_vector, dtype=np.float32)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.size or k <= 0:
            return []
        scores = self.cosines(query_vector)
        k = min(k, len(scores))
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
        return [(int(row), float(scores[row])) for row in order]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / VECTORS_NAME, self.vectors)
        with open(index_dir / CHUNKS_NAME, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        (index_dir / BM25_NAME).write_text(
            json.dumps(
                {"tokenizer": TOKENIZER_VERSION, "n_docs": len(self.records)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (index_dir / MANIFEST_NAME).write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "NumpyIndex | None":
        """读索引；缺文件或文件损坏返回 None（由调用方决定怎么报错）。"""
        if not index_dir.is_dir():
            return None
        if not all((index_dir / name).is_file() for name in REQUIRED_FILES):
            return None
        try:
            vectors = np.load(index_dir / VECTORS_NAME)
            records = [
                IndexRecord.from_dict(json.loads(line))
                for line in (index_dir / CHUNKS_NAME).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            manifest = json.loads((index_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"索引目录 {index_dir} 读取失败：{type(exc).__name__}: {exc}") from exc
        return cls(vectors, records, manifest)


def write_index_atomically(
    index_dir: Path,
    vectors: np.ndarray,
    records: list[IndexRecord],
    manifest: dict,
    report: dict,
) -> None:
    """先写临时目录、再整体替换，避免 agent 读到写了一半的索引。"""
    tmp_dir = index_dir.parent / f"{index_dir.name}.tmp"
    old_dir = index_dir.parent / f"{index_dir.name}.old"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(old_dir, ignore_errors=True)

    NumpyIndex(vectors, records, manifest).save(tmp_dir)
    (tmp_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    if index_dir.exists():
        index_dir.rename(old_dir)
    tmp_dir.rename(index_dir)
    shutil.rmtree(old_dir, ignore_errors=True)
