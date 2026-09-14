#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""体检：检查已下载语料是否完整、是否误把错误页存成了数据文件。

用法：python _scripts/validate.py [--show 40]
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from common import CORPUS, human, load_sources, setup_stdout

TEXT_EXT = {".json", ".jsonl", ".xml", ".csv", ".txt", ".md", ".html", ".htm"}
BAD_MARKERS = (
    b"<html",
    b"<!doctype html",
    b"Access Denied",
    b"403 Forbidden",
    b"404 Not Found",
    b"<Error>",
    b"no such file",
)
XML_SMALL = 12 * 1024 * 1024  # 12 MB 以内直接试解析


def xml_ok(path: Path, head: bytes) -> bool:
    snip = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if not snip.startswith(b"<"):
        return False
    if path.stat().st_size <= XML_SMALL:
        try:
            ET.parse(path)
            return True
        except ET.ParseError:
            return False
    # 大文件不整棵解析，只认合法的文档开头
    return snip.startswith((b"<?xml", b"<!DOCTYPE", b"<OAI-PMH", b"<DisorderList", b"<Orpha",
                            b"<mplus", b"<health-topics", b"<definition-page", b"<article"))


def suspicious(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_EXT:
        return None
    head = path.open("rb").read(4096).lstrip()
    low = head[:200].lower()
    for marker in BAD_MARKERS:
        if marker.lower() in low:
            if path.suffix.lower() in {".html", ".htm"}:
                continue  # 政府正文本来就是 HTML
            return f"疑似错误页（{marker.decode('ascii', 'ignore')}）"
    if path.suffix.lower() == ".xml" and not xml_ok(path, head):
        return "XML 头异常"
    return None


def main() -> int:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=40)
    args = parser.parse_args()

    rows = load_sources()
    ok = missing = empty = bad = 0
    problems: list[tuple[str, str]] = []
    total = 0
    for r in rows:
        path = CORPUS / r["local_path"]
        if not path.exists():
            if r.get("status") == "ok":
                missing += 1
                problems.append((r["local_path"], "登记为已完成但文件不存在"))
            continue
        size = path.stat().st_size
        total += size
        if size == 0:
            empty += 1
            problems.append((r["local_path"], "空文件"))
            continue
        issue = suspicious(path)
        if issue:
            bad += 1
            problems.append((r["local_path"], issue))
        else:
            ok += 1

    print(f"登记 {len(rows)} 条 | 存在且正常 {ok} | 缺失 {missing} | 空文件 {empty} | 可疑 {bad}")
    print(f"已落盘体积 {human(total)}")
    if problems:
        print(f"\n问题清单（前 {args.show} 条）：")
        for name, issue in problems[: args.show]:
            print(f"  {issue:<28} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
