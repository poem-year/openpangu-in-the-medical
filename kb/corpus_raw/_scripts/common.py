#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料采集公共工具：下载、哈希、来源登记表读写。

约定：

- 下载统一走 curl（Windows 自带，支持重试与断流保护），大文件可重复运行续跑。
- 每条来源登记在 corpus/sources.json，字段：
  id / category / source / license / url / local_path / note / status / bytes / sha256 / fetched_at
- 原始文件按类别落在 corpus/ 下的子目录，清单与脚本在 _scripts/，两者分开。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent
SOURCES = CORPUS / "sources.json"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def setup_stdout() -> None:
    """Windows 控制台按 UTF-8 输出，避免中文乱码。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def hf_url(repo: str, path: str) -> str:
    return f"{HF_ENDPOINT}/datasets/{repo}/resolve/main/{path}"


def hf_tree(repo: str) -> dict[str, int]:
    """一次拿到 HuggingFace 仓库的文件清单（path -> size）。"""
    import urllib.request

    url = f"{HF_ENDPOINT}/api/datasets/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        tree = json.load(resp)
    return {n["path"]: n.get("size") or 0 for n in tree if n.get("type") == "file"}


def load_sources() -> list[dict]:
    if SOURCES.exists():
        return json.loads(SOURCES.read_text(encoding="utf-8"))
    return []


LOCK = SOURCES.with_suffix(".lock")


def save_sources(rows: list[dict]) -> None:
    """加锁 + 原子替换写入，避免多进程采集时互相覆盖或写坏文件。"""
    rows = sorted(rows, key=lambda r: (r.get("category", ""), r.get("local_path", "")))
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    for attempt in range(4):
        tmp = SOURCES.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        try:
            os.replace(tmp, SOURCES)
            return
        except OSError:
            tmp.unlink(missing_ok=True)
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError("sources.json 写入失败（多次重试）")


class _Lock:
    """跨进程文件锁：sources.json 的读-改-写必须串行。"""

    def __enter__(self):
        for _ in range(240):
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - LOCK.stat().st_mtime > 120:
                        LOCK.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                time.sleep(0.5)
        raise RuntimeError("获取 sources.json 锁超时")

    def __exit__(self, *exc) -> None:
        LOCK.unlink(missing_ok=True)


def add_sources(rows: list[dict]) -> int:
    """按 local_path 去重登记，返回新增条数。"""
    with _Lock():
        cur = load_sources()
        by_path = {r["local_path"]: r for r in cur}
        added = 0
        for row in rows:
            path = row["local_path"]
            if path in by_path:
                # 保留已采集状态，只补齐来源字段
                for k, v in row.items():
                    by_path[path].setdefault(k, v)
                continue
            by_path[path] = row
            added += 1
        save_sources(list(by_path.values()))
        return added


def update_source(local_path: str, **fields) -> None:
    """只更新一条记录并立即落盘（多进程采集时避免互相覆盖）。"""
    with _Lock():
        rows = load_sources()
        for row in rows:
            if row["local_path"] == local_path:
                row.update(fields)
                break
        save_sources(rows)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(
    url: str,
    dest: Path,
    expect: int | None = None,
    retries: int = 3,
    timeout: int = 3600,
    referer: str | None = None,
    force: bool = False,
) -> str:
    """下载到 dest；已存在且大小一致则跳过。返回 ok / skip。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        if expect is None or dest.stat().st_size == expect:
            return "skip"
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "curl", "-sSL", "--http1.1", "--fail",
        "--retry", str(retries), "--retry-delay", "3", "--retry-all-errors",
        "--connect-timeout", "25", "--max-time", str(timeout),
        "--speed-limit", "1024", "--speed-time", "180",
        "-A", UA,
    ]
    if referer:
        cmd += ["-e", referer]
    cmd += ["-o", str(tmp), url]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"curl rc={proc.returncode} {proc.stderr.strip()[:200]}")
    size = tmp.stat().st_size
    if size == 0:
        tmp.unlink()
        raise RuntimeError("下载结果为空")
    if expect is not None and size != expect:
        tmp.unlink()
        raise RuntimeError(f"大小不符：{size} != {expect}")
    tmp.replace(dest)
    return "ok"


def fetch_text(url: str, timeout: int = 60, referer: str | None = None) -> str:
    """抓取一个小文本资源（HTML/JSON/XML），用于建库时的发现阶段。"""
    cmd = [
        "curl", "-sSL", "--http1.1", "--fail", "--retry", "2", "--retry-delay", "2",
        "--retry-all-errors", "--connect-timeout", "20", "--max-time", str(timeout),
        "-A", UA,
    ]
    if referer:
        cmd += ["-e", referer]
    cmd += [url]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"curl rc={proc.returncode} {proc.stderr.strip()[:200]}")
    return proc.stdout


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def entry(
    category: str,
    source: str,
    license_: str,
    url: str,
    local_path: str,
    note: str = "",
) -> dict:
    return {
        "id": local_path,
        "category": category,
        "source": source,
        "license": license_,
        "url": url,
        "local_path": local_path,
        "note": note,
        "status": "pending",
        "bytes": 0,
        "sha256": "",
        "fetched_at": "",
    }
