#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 sources.json：按类别 / 来源 / 许可统计语料规模与完成度。"""

from __future__ import annotations

import collections
from pathlib import Path

from common import CORPUS, human, load_sources, setup_stdout


def main() -> int:
    setup_stdout()
    rows = load_sources()
    done = [r for r in rows if r.get("status") == "ok" and (CORPUS / r["local_path"]).exists()]
    failed = [r for r in rows if r.get("status") == "failed"]

    print(f"登记 {len(rows)} 条 | 已完成 {len(done)} 条 | 失败 {len(failed)} 条")
    print(f"合计体积 {human(sum(r.get('bytes') or 0 for r in done))}")

    by_cat = collections.defaultdict(lambda: [0, 0])
    for r in done:
        by_cat[r["category"]][0] += 1
        by_cat[r["category"]][1] += r.get("bytes") or 0
    print("\n按类别：")
    for cat, (n, size) in sorted(by_cat.items()):
        print(f"  {cat:<18} {n:>3} 条  {human(size):>10}")

    by_src = collections.defaultdict(lambda: [0, 0])
    for r in done:
        by_src[r["source"]][0] += 1
        by_src[r["source"]][1] += r.get("bytes") or 0
    print("\n按来源（前 30）：")
    for src, (n, size) in sorted(by_src.items(), key=lambda x: -x[1][1])[:30]:
        print(f"  {src:<58} {n:>3} 条  {human(size):>10}")

    lic = collections.Counter(r["license"].split("（")[0] for r in done)
    print("\n按许可：")
    for name, n in lic.most_common():
        print(f"  {name:<24} {n:>3} 条")

    if failed:
        print("\n失败清单：")
        for r in failed:
            print("  ", r["local_path"], r["url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
