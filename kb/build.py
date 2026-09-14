"""建库 CLI：读语料 → 切块 → 嵌入 → 落盘（增量或全量）。

用法：
    .venv/bin/python -m kb.build              # 增量：按文件 sha1 跳过没变过的语料
    .venv/bin/python -m kb.build --rebuild    # 全量重建
    .venv/bin/python -m kb.build --corpus /path/to/corpus --index /path/to/index
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from kb.config import KBConfig, get_config
from kb.corpus import CHUNKER_VERSION, CorpusChunk, CorpusError, build_chunks
from kb.embed import EmbeddingBackend, create_backend
from kb.index import IndexRecord, NumpyIndex, write_index_atomically


def _to_record(chunk: CorpusChunk) -> IndexRecord:
    return IndexRecord(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        section=chunk.section,
        seq=chunk.seq,
        source_name=chunk.source_name,
        source_version=chunk.source_version,
        category=chunk.category,
        file_sha1=chunk.file_sha1,
        rel_path=chunk.rel_path,
        license=chunk.license,
        language=chunk.language,
        tier=chunk.tier,
    )


def build_index(
    config: KBConfig,
    *,
    backend: EmbeddingBackend | None = None,
    rebuild: bool = False,
    chunks: list[CorpusChunk] | None = None,
    log=print,
) -> tuple[dict, list[str]]:
    """建库。返回 (build_report, errors)。errors 非空时调用方应提示，但索引照常写出。"""
    started = time.time()
    if chunks is None:
        chunks, errors = build_chunks(config.corpus_dir, config)
    else:
        errors = []
    if not chunks:
        raise CorpusError(
            f"语料目录 {config.corpus_dir} 没有可入库的内容。\n"
            "请放入 md/txt/pdf 语料，并在 sources.yaml 登记后再跑。"
        )

    old = None
    if not rebuild:
        try:
            old = NumpyIndex.load(config.index_dir)
        except ValueError as exc:
            log(f"[build][警告] 旧索引读取失败，本次按全量重建：{exc}")
            old = None
    if old is not None and (old.manifest.get("embed_model") or config.embed_model) != config.embed_model:
        log(
            f"[build][警告] 嵌入模型变了（{old.manifest.get('embed_model')} → {config.embed_model}），"
            "旧向量不可复用，本次按全量重建"
        )
        old = None
    if old is not None and old.manifest.get("chunker_version") != CHUNKER_VERSION:
        log(
            f"[build][提示] 切块逻辑版本变了（{old.manifest.get('chunker_version')} → {CHUNKER_VERSION}），"
            "已入库的切块方式可能不同，本次按全量重建"
        )
        old = None

    old_row_by_id: dict[str, int] = {}
    if old is not None:
        old_row_by_id = {record.chunk_id: row for row, record in enumerate(old.records)}

    records = [_to_record(chunk) for chunk in chunks]
    duplicated = _find_duplicate_ids(records)
    if duplicated:
        raise CorpusError(
            "切块后出现重复 chunk_id（接口规范要求全局唯一）："
            + "、".join(duplicated[:5])
            + "。请检查是否有多份语料共用了同一个 doc_id。"
        )
    dim = old.dim if old is not None and old.dim else None
    vectors: list[np.ndarray | None] = []
    pending_rows: list[int] = []
    for row, record in enumerate(records):
        old_row = old_row_by_id.get(record.chunk_id)
        if old_row is not None and old is not None:
            vectors.append(old.vectors[old_row])
        else:
            vectors.append(None)
            pending_rows.append(row)

    reused = len(records) - len(pending_rows)
    n_files = len({record.rel_path for record in records})
    log(f"[build] 语料 {n_files} 份，切块 {len(records)} 条；复用向量 {reused} 条，待嵌入 {len(pending_rows)} 条")

    if pending_rows:
        embedder = backend or create_backend(config)
        dim = embedder.dim
        batch_size = max(1, config.embed_batch_size)
        t0 = time.time()
        done = 0
        for start in range(0, len(pending_rows), batch_size):
            batch_rows = pending_rows[start : start + batch_size]
            texts = [records[row].text for row in batch_rows]
            encoded = embedder.encode_documents(texts)
            for offset, row in enumerate(batch_rows):
                vectors[row] = encoded[offset]
            done += len(batch_rows)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (len(pending_rows) - done) / rate if rate > 0 else 0.0
            log(
                f"[build] 嵌入 {done}/{len(pending_rows)} "
                f"（{rate:.1f} 条/秒，已用 {elapsed:.0f}s，预计还需 {eta:.0f}s）"
            )
    elif dim is None:
        embedder = backend or create_backend(config)
        dim = embedder.dim

    matrix = np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])
    if dim is None:
        dim = int(matrix.shape[1])

    now = time.time()
    file_status = _file_status(old.manifest if old else {}, records)
    manifest = {
        "embed_model": config.embed_model,
        "chunker_version": CHUNKER_VERSION,
        "dim": int(dim),
        "n_chunks": len(records),
        "n_docs": len({record.doc_id for record in records}),
        "files": {record.rel_path: record.file_sha1 for record in records},
        "corpus_dir": str(config.corpus_dir),
        "built_at": now,
        "elapsed_seconds": round(now - started, 2),
    }
    report = {
        "built_at": now,
        "mode": "rebuild" if rebuild else "incremental",
        "embed_model": config.embed_model,
        "dim": int(dim),
        "n_files": n_files,
        "n_chunks": len(records),
        "n_docs": manifest["n_docs"],
        "reused_vectors": reused,
        "embedded_vectors": len(pending_rows),
        "removed_chunks": len(set(old_row_by_id) - {record.chunk_id for record in records}) if old else 0,
        "file_status": file_status,
        "errors": errors,
        "elapsed_seconds": round(now - started, 2),
    }
    write_index_atomically(config.index_dir, matrix, records, manifest, report)
    log(
        f"[build] 完成：{len(records)} 条切块、{manifest['n_docs']} 个文档，"
        f"耗时 {report['elapsed_seconds']}s，索引写入 {config.index_dir}"
    )
    return report, errors


def _file_status(old_manifest: dict, records: list[IndexRecord]) -> dict[str, str]:
    """按 sha1 对比新旧语料，标出每份文件是新增/变更/未变/已移除。"""
    old_files: dict[str, str] = dict(old_manifest.get("files") or {})
    new_files: dict[str, str] = {}
    for record in records:
        new_files[record.rel_path] = record.file_sha1
    status: dict[str, str] = {}
    for rel, sha in new_files.items():
        if rel not in old_files:
            status[rel] = "new"
        elif old_files[rel] != sha:
            status[rel] = "changed"
        else:
            status[rel] = "unchanged"
    for rel in old_files:
        if rel not in new_files:
            status[rel] = "removed"
    return status


def _find_duplicate_ids(records: list[IndexRecord]) -> list[str]:
    """找出重复的 chunk_id（正常应为一个不重复的集合）。"""
    seen: set[str] = set()
    duplicated: list[str] = []
    for record in records:
        if record.chunk_id in seen and record.chunk_id not in duplicated:
            duplicated.append(record.chunk_id)
        seen.add(record.chunk_id)
    return duplicated


def main() -> int:
    parser = argparse.ArgumentParser(description="建库：语料 → 切块 → 嵌入 → 索引")
    base = get_config()
    parser.add_argument("--corpus", type=Path, default=base.corpus_dir, help="语料目录")
    parser.add_argument("--index", type=Path, default=base.index_dir, help="索引输出目录")
    parser.add_argument("--rebuild", action="store_true", help="全量重建（默认为增量）")
    parser.add_argument("--model", default=base.embed_model, help="嵌入模型名")
    parser.add_argument("--device", default=base.device, help="嵌入设备（cpu）")
    parser.add_argument("--batch-size", type=int, default=base.embed_batch_size)
    parser.add_argument(
        "--dump-chunks",
        type=Path,
        help="只切块并写出 JSONL（供另一个环境嵌入，见 kb/README.md 两段式建库）",
    )
    parser.add_argument("--load-chunks", type=Path, help="跳过切块，直接读 JSONL 里的块做嵌入")
    args = parser.parse_args()

    config = KBConfig(
        corpus_dir=args.corpus,
        index_dir=args.index,
        embed_model=args.model,
        device=args.device,
        embed_batch_size=args.batch_size,
    ).validated()

    if args.dump_chunks:
        from dataclasses import asdict

        chunks, errors = build_chunks(config.corpus_dir, config)
        args.dump_chunks.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_chunks.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        print(f"[build] 切块 {len(chunks)} 条 → {args.dump_chunks}；错误 {len(errors)} 条")
        for message in errors:
            print(f"[build][错误] {message}")
        return 1 if errors else 0

    preloaded: list[CorpusChunk] | None = None
    if args.load_chunks:
        preloaded = []
        with args.load_chunks.open(encoding="utf-8") as fh:
            for line in fh:
                preloaded.append(CorpusChunk(**json.loads(line)))
        print(f"[build] 从 {args.load_chunks} 读入切块 {len(preloaded)} 条")

    try:
        report, errors = build_index(config, rebuild=args.rebuild, chunks=preloaded)
    except CorpusError as exc:
        print(f"[build] 建库失败：{exc}")
        return 2

    for message in errors:
        print(f"[build][错误] {message}")
    print("[build] 结果明细：" + json.dumps(report["file_status"], ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
