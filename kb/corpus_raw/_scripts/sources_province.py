#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""爬取省级卫健委 / 疾控中心官网的公开文件与健康指南。

做法：两跳爬取——首页找"政策文件 / 通知公告 / 健康科普"栏目页，再从栏目页
捞正文链接；标题命中医学关键词才登记。限定域名白名单与总量，避免跑飞。

用法：python _scripts/sources_province.py [--per-site 80]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import urllib.parse

from common import add_sources, entry, fetch_text, setup_stdout

SITES = [
    ("上海市卫生健康委员会", "https://wsjkw.sh.gov.cn/"),
    ("广东省卫生健康委员会", "http://wsjkw.gd.gov.cn/"),
    ("浙江省卫生健康委员会", "https://wsjkw.zj.gov.cn/"),
    ("湖北省卫生健康委员会", "http://wjw.hubei.gov.cn/"),
    ("中国疾病预防控制中心", "https://www.chinacdc.cn/"),
]

SECTION_HINT = ("政策", "文件", "通知", "公告", "指南", "规范", "标准", "方案", "法规", "健康", "科普", "疾病")
TITLE_HINT = ("诊疗", "防治", "指南", "规范", "标准", "方案", "共识", "诊断", "疾病", "传染病",
              "慢病", "免疫", "疫苗", "急诊", "急救", "健康", "临床", "用药", "检查")
SKIP_EXT = (".jpg", ".png", ".gif", ".mp4", ".zip", ".rar")


def links(html: str, base: str) -> list[tuple[str, str]]:
    out = []
    for m in re.finditer(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", html, re.I | re.S):
        href, text = m.group(1), re.sub(r"<[^>]+>", "", m.group(2))
        text = re.sub(r"\s+", " ", text).strip()
        if not href or href.startswith(("javascript:", "#", "mailto:")):
            continue
        url = urllib.parse.urljoin(base, href)
        if any(url.lower().endswith(ext) for ext in SKIP_EXT):
            continue
        out.append((url, text))
    return out


def main() -> int:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-site", type=int, default=80)
    args = parser.parse_args()

    rows: list[dict] = []
    seen_url: set[str] = set()
    for name, home in SITES:
        try:
            html = fetch_text(home, timeout=45)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 首页失败 {name}: {exc}")
            continue
        sections = [u for u, t in links(html, home) if any(h in t for h in SECTION_HINT)]
        sections = list(dict.fromkeys(sections))[:12]
        count = 0
        for sec in sections:
            if count >= args.per_site:
                break
            try:
                sec_html = fetch_text(sec, timeout=45)
            except Exception:
                continue
            for url, text in links(sec_html, sec):
                if count >= args.per_site:
                    break
                if url in seen_url or url.rstrip("/") == home.rstrip("/"):
                    continue
                if not any(h in text for h in TITLE_HINT):
                    continue
                if not re.search(r"(19|20)\d{2}", url) and not re.search(r"\d{5,}", url):
                    continue
                seen_url.add(url)
                ident = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
                title = re.sub(r"[\\/:*?\"<>|\s]+", "_", text)[:60] or "doc"
                rows.append(
                    entry(
                        category="guidelines_cn",
                        source=name,
                        license_="政府/事业单位公开文件（注明出处）",
                        url=url,
                        local_path=f"guidelines_cn/province/{name}/{title}_{ident}.html",
                        note=f"栏目页：{sec}",
                    )
                )
                count += 1
        print(f"[info] {name}: 登记 {count} 条")

    added = add_sources(rows)
    print(f"[ok] 省级/疾控站点合计登记 {len(rows)} 条，新增 {added} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
