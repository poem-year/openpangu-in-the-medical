#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集魔搭（ModelScope）社区里的中文医学数据集。

只收许可明确（Apache-2.0 / MIT）的仓库，逐个仓库设体积上限，避免一次拉爆。

用法：python _scripts/sources_modelscope.py [--max-gb 5]
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request

from common import UA, add_sources, entry, setup_stdout

TREE = "https://www.modelscope.cn/api/v1/datasets/{repo}/repo/tree?Revision=master&Recursive=true"
FILE = "https://www.modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={path}"

# (仓库, 许可, 说明, 体积上限 MB, 只取包含该子串的路径)
DATASETS: list[tuple[str, str, str, int, str]] = [
    ("BAAI/IndustryCorpus_medicine", "Apache-2.0", "中文医学领域语料（BAAI 行业语料医学分册）", 1600, "/"),
    ("BAAI/IndustryInstruction_Health-Medicine", "Apache-2.0", "健康医疗指令数据（BAAI）", 500, ""),
    ("SHPDataGR/medical-exam-question-bank-json", "Apache-2.0", "医学考试题库（JSON）", 300, ""),
    ("BRZ911/Medical_consultation_data_SFT", "Apache-2.0", "医疗咨询对话 SFT 数据", 500, ""),
    ("krisfu/delicate_medical_r1_data", "Apache-2.0", "医学推理（R1 蒸馏）数据", 500, ""),
    ("huangxp/hwtcm-deepseek-r1-distill-data", "Apache-2.0", "中医推理蒸馏数据", 400, ""),
    ("huangxp/hwtcm-sft-v1", "Apache-2.0", "中医 SFT 语料", 300, ""),
    ("xiaofengalg/ShenNong_TCM_Dataset", "Apache-2.0", "神农中医数据集（问答/药材）", 300, ""),
    ("joshuaHe/tcm_pretrain_corpus", "Apache-2.0", "中医预训练语料", 500, ""),
    ("chuanqi/CBLUE-KUAKE-QQR", "Apache-2.0", "中文医疗问答（CBLUE 酷客）", 200, ""),
    ("qyhhhhhhhhh/medopd", "Apache-2.0", "医学开放问诊数据", 300, ""),
    ("alexhuangguo/chinese-medical", "Apache-2.0", "中文医学语料", 400, ""),
]

SKIP_NAME = (".gitattributes", "README", "dataset_infos.json", ".gitignore")


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def main() -> int:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-gb", type=float, default=5.0)
    args = parser.parse_args()

    budget = args.max_gb * 1024
    rows: list[dict] = []
    used_total = 0.0

    for repo, license_, note, cap_mb, contains in DATASETS:
        if used_total >= budget:
            print("[info] 已达总体积上限，停止登记")
            break
        try:
            files = (get_json(TREE.format(repo=repo)).get("Data") or {}).get("Files") or []
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 取清单失败 {repo}: {exc}")
            continue
        picked, used = [], 0.0
        for f in sorted(files, key=lambda x: x.get("Size") or 0):
            path = f.get("Path") or ""
            size = f.get("Size") or 0
            if not size or f.get("Type") != "blob":
                continue
            if any(s in path for s in SKIP_NAME):
                continue
            if contains and contains not in path:
                continue
            if used + size / 1e6 > min(cap_mb, budget - used_total):
                continue
            picked.append((path, size))
            used += size / 1e6
        for path, size in picked:
            name = path.replace("/", "__")
            rows.append(
                entry(
                    category="ms_datasets",
                    source=f"ModelScope（魔搭）：{repo}",
                    license_=license_,
                    url=FILE.format(repo=repo, path=urllib.parse.quote(path)),
                    local_path=f"hf_datasets/modelscope/{repo.replace('/', '__')}/{name}",
                    note=f"{note}（{size / 1e6:.1f} MB）",
                )
            )
        used_total += used
        print(f"[info] {repo}: 选中 {len(picked)} 个文件，{used:.1f} MB（累计 {used_total:.1f} MB）")

    added = add_sources(rows)
    print(f"[ok] 魔搭登记 {len(rows)} 个文件，新增 {added} 条，约 {used_total / 1024:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
