#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""登记 Orphadata（Orphanet 开放数据，罕见病）——许可 CC BY 4.0。

用途：D08 疑难/罕见病例、鉴别诊断的知识底座。
"""

from __future__ import annotations

from common import add_sources, entry, setup_stdout

BASE = "https://www.orphadata.com/data/xml"

FILES = [
    ("en_product1.xml", "罕见病命名与分类（Orphanet 本体）"),
    ("en_product4.xml", "罕见病—表型关联（HPO 表型）"),
    ("en_product6.xml", "罕见病相关基因"),
    ("en_product9_ages.xml", "罕见病发病年龄"),
    ("en_product9_prev.xml", "罕见病流行病学（患病率/发病数）"),
]


def main() -> int:
    setup_stdout()
    rows = [
        entry(
            category="guidelines_intl",
            source="Orphadata (Orphanet)",
            license_="CC-BY-4.0",
            url=f"{BASE}/{name}",
            local_path=f"guidelines_intl/orphadata/{name}",
            note=note,
        )
        for name, note in FILES
    ]
    added = add_sources(rows)
    print(f"[ok] Orphadata 登记 {len(rows)} 条，新增 {added} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
