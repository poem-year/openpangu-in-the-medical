#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集中文维基百科的医学条目（CC BY-SA 4.0，需署名+相同方式共享）。

用法：python _scripts/sources_wikipedia.py [--max 800]
产物：encyclopedia/wikipedia_zh/zhwiki_medical.jsonl（一条=一个条目）
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from common import CORPUS, setup_stdout

API = "https://zh.wikipedia.org/w/api.php"
OUT = CORPUS / "encyclopedia" / "wikipedia_zh" / "zhwiki_medical.jsonl"

ROOT_CATEGORIES = ["疾病", "症状", "医学检查", "医学影像", "外科手术", "传染病"]

# 维基媒体要求可识别的 UA；429 时按指数退避重试
WIKI_UA = "openpangu-medical-kb/0.1 (research corpus; contact: openpangu@local)"


def api(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    delay = 5
    for attempt in range(4):
        req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA, "Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                import gzip

                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < 3:
                print(f"[warn] {exc.code} 限流，等 {delay}s 重试")
                time.sleep(delay)
                delay *= 3
                continue
            raise
        finally:
            time.sleep(1.2)
    raise RuntimeError("重试后仍失败")


def category_members(cat: str, limit: int = 500) -> tuple[list[str], list[str]]:
    """返回 (条目名, 子分类名)。"""
    pages, subs = [], []
    cont = None
    while len(pages) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{cat}",
            "cmlimit": "500",
        }
        if cont:
            params["cmcontinue"] = cont
        data = api(params)
        for m in data.get("query", {}).get("categorymembers", []):
            if m.get("ns") == 0:
                pages.append(m["title"])
            elif m.get("ns") == 14:
                subs.append(m["title"].split(":", 1)[-1])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return pages, subs


def main() -> int:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=800)
    args = parser.parse_args()

    titles: list[str] = []
    seen: set[str] = set()
    for cat in ROOT_CATEGORIES:
        try:
            pages, subs = category_members(cat)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 分类失败 {cat}: {exc}")
            continue
        for t in pages:
            if t not in seen:
                seen.add(t)
                titles.append(t)
        # 一级子分类再取一轮，扩大覆盖面
        for sub in subs[:6]:
            if len(titles) >= args.max:
                break
            try:
                more, _ = category_members(sub, limit=120)
            except Exception:
                continue
            for t in more:
                if t not in seen:
                    seen.add(t)
                    titles.append(t)
        print(f"[info] 分类 {cat}: 累计 {len(titles)} 个条目")
        if len(titles) >= args.max:
            break

    titles = titles[: args.max]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i in range(0, len(titles), 20):
            batch = titles[i : i + 20]
            try:
                data = api(
                    {
                        "action": "query",
                        "prop": "extracts|revisions|info",
                        "explaintext": "1",
                        "exsectionformat": "plain",
                        "rvprop": "timestamp|ids",
                        "inprop": "url",
                        "titles": "|".join(batch),
                        "redirects": "1",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 正文失败 {batch[:2]}...: {exc}")
                continue
            for page in data.get("query", {}).get("pages", []):
                text = page.get("extract") or ""
                if len(text) < 500:
                    continue
                rev = (page.get("revisions") or [{}])[0]
                f.write(
                    json.dumps(
                        {
                            "title": page.get("title"),
                            "pageid": page.get("pageid"),
                            "url": page.get("fullurl"),
                            "revision": rev.get("revid"),
                            "timestamp": rev.get("timestamp"),
                            "extract": text,
                            "source": "中文维基百科（CC BY-SA 4.0）",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
            time.sleep(0.3)
    print(f"[ok] 中文维基医学条目 {written} 条 -> {OUT.relative_to(CORPUS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
