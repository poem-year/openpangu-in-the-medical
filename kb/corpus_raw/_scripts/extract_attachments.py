#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从已下载的政府文件正文里提取附件链接（PDF/Word/Excel），登记为待下载条目。

用法：python _scripts/extract_attachments.py
"""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

from common import CORPUS, add_sources, entry, load_sources, setup_stdout

EXT_RE = re.compile(r"href=[\"']([^\"']+\.(?:pdf|docx?|xlsx?|wps))[\"']", re.I)


def main() -> int:
    setup_stdout()
    rows = load_sources()
    new_rows: list[dict] = []
    seen = {r["local_path"] for r in rows}
    for r in rows:
        if r["category"] != "guidelines_cn" or not r["local_path"].endswith(".html"):
            continue
        path = CORPUS / r["local_path"]
        if not path.exists():
            continue
        try:
            html = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        base = r["url"]
        for href in set(EXT_RE.findall(html)):
            url = urllib.parse.urljoin(base, href)
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
            name = re.sub(r"[\\/:*?\"<>|\s]+", "_", name)[:80]
            local = f"{Path(r['local_path']).parent}/{name}"
            if local in seen:
                continue
            seen.add(local)
            new_rows.append(
                entry(
                    category="guidelines_cn",
                    source=r["source"] + "（附件）",
                    license_=r["license"],
                    url=url,
                    local_path=local,
                    note=f"来自正文：{r['local_path']}",
                )
            )
    added = add_sources(new_rows)
    print(f"[ok] 提取附件 {len(new_rows)} 条，新增 {added} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
