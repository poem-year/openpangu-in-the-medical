#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集中国政府网政策文件库里的医疗卫生文件（政府文件不受著作权保护）。

做法：调 gov.cn 的公开检索接口（sousuo.www.gov.cn），按医学关键词检索
国务院文件与部门文件，登记正文页 URL；正文里的 PDF/Word 附件由
extract_attachments.py 二次提取。

用法：python _scripts/sources_gov_cn.py [--per-query 15] [--pages 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request

from common import UA, add_sources, entry, setup_stdout

API = "https://sousuo.www.gov.cn/search-gov/data"

# 与诊断能力直接相关的检索词（诊疗方案 / 诊断标准 / 检查决策 / 安全边界）
QUERIES = [
    "诊疗方案", "诊疗指南", "诊断标准", "临床路径", "技术指南",
    "防控方案", "防治指南", "专家共识", "疾病分类与代码", "分级诊疗",
    "慢性病防治", "免疫规划", "传染病防治", "急诊急救", "胸痛中心",
    "卒中中心", "肿瘤诊疗", "孕产妇保健", "儿童保健", "医院感染",
]

TYPES = ["zhengcelibrary_gw", "zhengcelibrary_bm"]


def search(query: str, page: int, n: int, type_: str) -> list[dict]:
    params = {"t": type_, "q": query, "p": page, "n": n}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    return (data.get("searchVO") or {}).get("listVO") or []


def safe_name(title: str, ident: str) -> str:
    title = re.sub(r"<[^>]+>", "", title or "")
    title = re.sub(r"[\\/:*?\"<>|\s]+", "_", title).strip("_")
    return f"{title[:60]}_{ident}.html"


def main() -> int:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query", type=int, default=15)
    parser.add_argument("--pages", type=int, default=2)
    args = parser.parse_args()

    rows, seen = [], set()
    for type_ in TYPES:
        for query in QUERIES:
            for page in range(1, args.pages + 1):
                try:
                    items = search(query, page, args.per_query, type_)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] {type_} {query} p{page}: {exc}")
                    continue
                for it in items:
                    url = it.get("url") or it.get("piclinksurl") or ""
                    if not url.startswith("https://www.gov.cn"):
                        continue
                    if url in seen:
                        continue
                    seen.add(url)
                    ident = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
                    rows.append(
                        entry(
                            category="guidelines_cn",
                            source="中国政府网（gov.cn）政策文件库",
                            license_="政府文件（不受著作权保护）",
                            url=url,
                            local_path=f"guidelines_cn/gov_cn/{safe_name(it.get('title', ''), ident)}",
                            note=f"检索词：{query}；发文机关：{it.get('puborg') or '-'}；发布日期：{it.get('pubtimeStr') or '-'}",
                        )
                    )
    added = add_sources(rows)
    print(f"[ok] gov.cn 登记 {len(rows)} 条，新增 {added} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
