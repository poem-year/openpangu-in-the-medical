#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集 Europe PMC 的开放获取全文（只取 CC BY 许可的文章）。

为什么选它：这是能合法批量拿全文综述/指南的正路——按许可过滤、按被引排序、
带 DOI 与期刊信息，适合做诊断类知识库的"证据层"。

用法：python _scripts/sources_europepmc.py [--per-query 25] [--max 600]
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from common import CORPUS, UA, add_sources, entry, setup_stdout

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC/{pmcid}/fullTextXML"
INDEX = CORPUS / "literature_oa" / "europepmc" / "_index.jsonl"

# 按测评集的能力方向组织的检索词：诊断 / 鉴别 / 检查 / 急症 / 沟通
QUERIES = [
    "acute coronary syndrome diagnosis",
    "heart failure diagnosis guideline",
    "atrial fibrillation diagnosis",
    "community acquired pneumonia diagnosis",
    "pulmonary embolism diagnosis",
    "deep vein thrombosis diagnosis",
    "sepsis early recognition",
    "acute stroke thrombolysis",
    "diabetes mellitus diagnostic criteria",
    "hypertension diagnosis guideline",
    "chronic kidney disease staging",
    "thyroid nodule evaluation",
    "breast cancer diagnostic pathway",
    "colorectal cancer screening diagnosis",
    "hepatitis B diagnosis management",
    "tuberculosis diagnosis",
    "meningitis diagnosis differential",
    "fever of unknown origin workup",
    "anemia diagnostic workup",
    "syncope differential diagnosis",
    "chest pain triage emergency",
    "abdominal pain differential diagnosis",
    "preeclampsia diagnosis management",
    "pediatric fever evaluation",
    "rare disease diagnostic delay",
    "diagnostic error patient safety",
    "shared decision making communication diagnosis",
    "clinical decision support diagnosis accuracy",
]


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def main() -> int:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query", type=int, default=25)
    parser.add_argument("--max", type=int, default=600)
    args = parser.parse_args()

    rows: list[dict] = []
    meta_rows: list[dict] = []
    seen: set[str] = set()

    for query in QUERIES:
        if len(rows) >= args.max:
            break
        expr = f'({query}) AND OPEN_ACCESS:Y AND LICENSE:"cc by" AND HAS_FT:Y'
        params = {
            "query": expr,
            "format": "json",
            "pageSize": args.per_query,
            "resultType": "core",
            "sort": "CITED desc",
        }
        try:
            data = get_json(SEARCH + "?" + urllib.parse.urlencode(params))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 检索失败 {query}: {exc}")
            continue
        hits = (data.get("resultList") or {}).get("result") or []
        picked = 0
        for hit in hits:
            pmcid = hit.get("pmcid")
            if not pmcid or pmcid in seen or len(rows) >= args.max:
                continue
            seen.add(pmcid)
            rows.append(
                entry(
                    category="literature_oa",
                    source=f"Europe PMC（期刊：{hit.get('journalInfo', {}).get('journal', {}).get('title') or '未知'}）",
                    license_=f"CC BY（{hit.get('license') or 'cc by'}）",
                    url=FULLTEXT.format(pmcid=pmcid),
                    local_path=f"literature_oa/europepmc/{pmcid}.xml",
                    note=f"检索词：{query}；标题：{str(hit.get('title'))[:120]}",
                )
            )
            meta_rows.append(
                {
                    "pmcid": pmcid,
                    "pmid": hit.get("pmid"),
                    "doi": hit.get("doi"),
                    "title": hit.get("title"),
                    "journal": (hit.get("journalInfo") or {}).get("journal", {}).get("title"),
                    "year": hit.get("pubYear"),
                    "license": hit.get("license"),
                    "query": query,
                }
            )
            picked += 1
        print(f"[info] {query}: 取 {picked} 篇（累计 {len(rows)}）")

    if meta_rows:
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        with INDEX.open("a", encoding="utf-8") as f:
            for m in meta_rows:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    added = add_sources(rows)
    print(f"[ok] Europe PMC 登记 {len(rows)} 篇，新增 {added} 条；索引 -> {INDEX.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
