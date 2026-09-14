#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""登记 MedlinePlus 健康主题（美国国家医学图书馆）。

许可：混合。NLM 自产内容多为公有领域，第三方授权内容受版权保护；本采集只取
官方发布的聚合 XML，并在 manifest 中标注"混合许可，逐条核对后再对外发布"。

用法：python _scripts/sources_medlineplus.py
"""

from __future__ import annotations

import re

from common import add_sources, entry, fetch_text, setup_stdout

PAGE = "https://medlineplus.gov/xml.html"


def latest(pattern: str, html: str) -> str | None:
    hits = sorted(set(re.findall(pattern, html)), reverse=True)
    return hits[0] if hits else None


def main() -> int:
    setup_stdout()
    html = fetch_text(PAGE)
    rows = []

    topics = latest(r"https://medlineplus\.gov/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml", html)
    if topics:
        rows.append(
            entry(
                category="guidelines_intl",
                source="MedlinePlus (NLM)",
                license_="混合（NLM 公有领域 + 第三方授权，仅内部研究）",
                url=topics,
                local_path=f"guidelines_intl/medlineplus/{topics.rsplit('/', 1)[-1]}",
                note="健康主题全文 XML（约 1000 个主题），患者语言层",
            )
        )

    groups = latest(r"https://medlineplus\.gov/xml/mplus_topic_groups_\d{4}-\d{2}-\d{2}\.xml", html)
    if groups:
        rows.append(
            entry(
                category="guidelines_intl",
                source="MedlinePlus (NLM)",
                license_="混合（仅内部研究）",
                url=groups,
                local_path=f"guidelines_intl/medlineplus/{groups.rsplit('/', 1)[-1]}",
                note="健康主题分组结构",
            )
        )

    for name in ("fitnessdefinitions", "generalhealthdefinitions", "nutritiondefinitions",
                 "vitaminsdefinitions", "mineralsdefinitions"):
        url = f"https://medlineplus.gov/xml/{name}.xml"
        rows.append(
            entry(
                category="guidelines_intl",
                source="MedlinePlus (NLM)",
                license_="混合（仅内部研究）",
                url=url,
                local_path=f"guidelines_intl/medlineplus/{name}.xml",
                note="健康术语定义",
            )
        )

    added = add_sources([r for r in rows if r])
    for r in rows:
        if r:
            print("  ", r["local_path"], "<-", r["url"])
    print(f"[ok] MedlinePlus 登记 {len(rows)} 条，新增 {added} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
