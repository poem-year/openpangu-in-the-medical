#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 sources.json 下载语料，支持断点续跑与状态回写。

用法：

    python _scripts/download.py                    # 下载全部未完成条目
    python _scripts/download.py --category hf_datasets
    python _scripts/download.py --limit 5 --dry-run
    python _scripts/download.py --force            # 重下（忽略已有文件）

每条下载完成后回写 status / bytes / sha256 / fetched_at 到 sources.json。
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from common import CORPUS, download, human, load_sources, setup_stdout, sha256, update_source


def main() -> int:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="只下某一类（如 hf_datasets）")
    parser.add_argument("--limit", type=int, default=0, help="最多下载几条")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-bytes", type=int, default=0, help="只为小于该大小的条目补哈希")
    args = parser.parse_args()

    rows = load_sources()
    todo = [r for r in rows if args.category is None or r["category"] == args.category]
    pending = []
    for r in todo:
        dest = CORPUS / r["local_path"]
        if r.get("status") == "ok" and dest.exists() and not args.force:
            continue
        pending.append(r)
    if args.limit:
        pending = pending[: args.limit]

    print(f"[plan] 待处理 {len(pending)} 条（总登记 {len(rows)} 条）")
    if args.dry_run:
        for r in pending:
            print("  ", r["local_path"], r["source"])
        return 0

    ok = fail = skip = 0
    total_bytes = 0
    for i, r in enumerate(pending, 1):
        dest = CORPUS / r["local_path"]
        prefix = f"[{i}/{len(pending)}]"
        try:
            result = download(r["url"], dest, force=args.force)
        except Exception as exc:  # noqa: BLE001
            update_source(r["local_path"], status="failed")
            fail += 1
            print(f"{prefix} 失败 {r['local_path']}: {exc}", file=sys.stderr)
            continue
        size = dest.stat().st_size
        fields = {
            "status": "ok",
            "bytes": size,
            "fetched_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if size <= 200 * 1024 * 1024:
            fields["sha256"] = sha256(dest)
        update_source(r["local_path"], **fields)
        total_bytes += size
        if result == "skip":
            skip += 1
        else:
            ok += 1
        print(f"{prefix} {result:<4} {human(size):>10}  {r['local_path']}")

    print(f"[done] 新增 {ok}，跳过 {skip}，失败 {fail}，本轮 {human(total_bytes)}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
