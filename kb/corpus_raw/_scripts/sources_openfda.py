#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""登记 openFDA 药品说明书批量数据（美国 FDA，公有领域）。

策略：读 openFDA 的 download.json 清单，取 drug/label 的若干分片，控制在
--max-gb 以内（默认 1.5 GB），避免一次拉几十 GB。

用法：python _scripts/sources_openfda.py [--max-gb 1.5]
"""

from __future__ import annotations

import argparse
import json

from common import add_sources, entry, fetch_text, setup_stdout

INDEX = "https://api.fda.gov/download.json"


def main() -> int:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-gb", type=float, default=1.5)
    args = parser.parse_args()

    data = json.loads(fetch_text(INDEX, timeout=90))
    label = data["results"]["drug"]["label"]
    partitions = label["partitions"]
    print(f"[info] drug/label 共 {len(partitions)} 个分片，总计 {label.get('total_records')} 条记录")

    budget = args.max_gb * 1024
    picked, used = [], 0.0
    for part in sorted(partitions, key=lambda p: p.get("size_mb") or 0):
        size = float(part.get("size_mb") or 0)
        if used + size > budget:
            continue
        picked.append(part)
        used += size
    picked.sort(key=lambda p: p["file"])

    rows = [
        entry(
            category="drug_labels",
            source="openFDA (US FDA)",
            license_="公有领域（美国政府作品）",
            url=p["file"],
            local_path=f"drug_labels/openfda/{p['file'].rsplit('/', 1)[-1]}",
            note=f"药品说明书批量数据（{p.get('records')} 条，{p.get('size_mb')} MB）",
        )
        for p in picked
    ]
    added = add_sources(rows)
    print(f"[ok] openFDA 选中 {len(rows)} 个分片、约 {used / 1024:.2f} GB，新增 {added} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
