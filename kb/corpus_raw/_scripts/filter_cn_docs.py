#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把政府网检索结果里与医学无关的文件移到 _excluded/（可逆，只移动不删除）。

背景：gov.cn 检索接口对"诊疗方案"这类词会返回宽松匹配结果，混进快递、林业、
商务类文件。这里按标题关键词做一次粗筛，非医学文件移出主目录并标记 excluded。

用法：python _scripts/filter_cn_docs.py [--apply]
"""

from __future__ import annotations

import argparse
import shutil

from common import CORPUS, load_sources, setup_stdout, update_source

MUST_HAVE = [
    "医", "药", "病", "健康", "卫生", "护理", "康复", "疫苗", "传染", "疾控",
    "中医", "营养", "妇幼", "急救", "临床", "诊断", "诊疗", "心理", "精神",
    "血液", "器官", "职业健康", "公共卫生", "残疾", "医保", "医疗",
]


def main() -> int:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="真正移动文件（默认只统计）")
    args = parser.parse_args()

    rows = load_sources()
    keep = drop = 0
    for r in rows:
        if r["category"] != "guidelines_cn" or r.get("status") != "ok":
            continue
        text = r["local_path"] + " " + r.get("note", "")
        if any(k in text for k in MUST_HAVE):
            keep += 1
            continue
        drop += 1
        src = CORPUS / r["local_path"]
        if args.apply:
            if src.exists():
                dest = CORPUS / "guidelines_cn" / "_excluded" / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest.unlink()
                shutil.move(str(src), str(dest))
            update_source(r["local_path"], status="excluded")
        else:
            print("  待剔除：", r["local_path"])
    print(f"[ok] 医学相关 {keep} 条，非医学 {drop} 条" + ("（已移动）" if args.apply else "（试运行，加 --apply 生效）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
