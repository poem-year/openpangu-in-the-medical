#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把各权威公开集的原始文件下载到 _raw/。

只用 HuggingFace 镜像（环境变量 HF_ENDPOINT，默认 hf-mirror.com），不依赖
huggingface_hub。已下载且大小一致的文件会跳过，可重复运行。
"""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "_raw"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

# (本地子目录, 仓库 id, 仓库内路径)
FILES = [
    # 中文 · 权威考试与临床病例
    ("CMB", "FreedomIntelligence/CMB", "CMB-Clin/CMB-Clin-qa.json"),
    ("CMB", "FreedomIntelligence/CMB", "CMB-Exam/CMB-test/CMB-test-choice-question-merge.json"),
    ("CMB", "FreedomIntelligence/CMB", "CMB-Exam/CMB-val/CMB-val-merge.json"),
    ("CMExam", "fzkuji/CMExam", "test.json"),
    ("CMExam", "fzkuji/CMExam", "valid.json"),
    ("PromptCBLUE", "tchenglv/PromptCBLUE", "dev.json"),
    # 英文 · 权威考试
    ("MMLU", "cais/mmlu", "clinical_knowledge/test-00000-of-00001.parquet"),
    ("MMLU", "cais/mmlu", "medical_genetics/test-00000-of-00001.parquet"),
    ("MMLU", "cais/mmlu", "professional_medicine/test-00000-of-00001.parquet"),
    ("MMLU", "cais/mmlu", "college_medicine/test-00000-of-00001.parquet"),
    ("MMLU", "cais/mmlu", "anatomy/test-00000-of-00001.parquet"),
    ("MedQA", "bigbio/med_qa", "med_qa_zh_4options_bigbio_qa/test-00000-of-00001.parquet"),
    ("MedQA", "bigbio/med_qa", "med_qa_en_4options_bigbio_qa/test-00000-of-00001.parquet"),
    ("MedMCQA", "openlifescienceai/medmcqa", "data/validation-00000-of-00001.parquet"),
    # 英文 · 真实临床病例与端到端
    ("NEJM_CPC", "katielink/nejm-medqa-diagnostic-reasoning-dataset",
     "nejm_test/train-00000-of-00001.parquet"),
    ("NEJM_CPC", "katielink/nejm-medqa-diagnostic-reasoning-dataset",
     "41746_2024_1010_MOESM2_ESM.csv"),
    ("NEJM_CPC", "katielink/nejm-medqa-diagnostic-reasoning-dataset",
     "41746_2024_1010_MOESM4_ESM.csv"),
    ("MedXpertQA", "TsinghuaC3I/MedXpertQA", "Text/test.jsonl"),
    ("MedXpertQA", "TsinghuaC3I/MedXpertQA", "Text/dev.jsonl"),
    # 多轮问诊 / 智能体
    ("AgentClinic", "katielink/agentclinic_medqa", "agentclinic_medqa.jsonl"),
    # 安全边界与真实场景
    ("HealthBench", "openai/healthbench", "2025-05-07-06-14-12_oss_eval.jsonl"),
    ("HealthBench", "openai/healthbench", "hard_2025-05-08-21-00-10.jsonl"),
    ("MedBenchSafety", "miugod/MedBench-Safety-Ethics", "data/train-00000-of-00001.parquet"),
    # 临床计算
    ("MedCalcBench", "ncbi/MedCalc-Bench", "test_data_11_18_final.csv"),
    # 多模态
    ("SLAKE", "BoKelvin/SLAKE", "test.json"),
    ("SLAKE", "BoKelvin/SLAKE", "imgs.zip"),
    ("VQARAD", "flaviagiammarino/vqa-rad", "data/test-00000-of-00001-e5bc3d208bb4deeb.parquet"),
    ("PathVQA", "flaviagiammarino/path-vqa", "data/test-00000-of-00003-e9adadb4799f44d3.parquet"),
    # 证据检索
    ("PubMedQA", "qiaojin/PubMedQA", "pqa_labeled/train-00000-of-00001.parquet"),
    ("CmedqaRetrieval", "C-MTEB/CmedqaRetrieval", "data/queries-00000-of-00001-daeedab899d3c839.parquet"),
    ("CmedqaRetrieval", "C-MTEB/CmedqaRetrieval", "data/corpus-00000-of-00001-a3949861f65a3226.parquet"),
]


def repo_tree(repo: str) -> dict[str, int]:
    """一次拿到整个仓库的文件清单，避免逐文件查询。"""
    url = f"{ENDPOINT}/api/datasets/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    try:
        # 镜像会拦 Python 默认 UA，这里借用 curl 的 UA。
        with urllib.request.urlopen(req, timeout=30) as resp:
            tree = json.load(resp)
    except Exception:
        return {}
    return {
        node["path"]: node.get("size")
        for node in tree
        if node.get("type") == "file"
    }


def download(repo: str, path: str, dest: Path, expect: int | None, retries: int = 3) -> bool:
    if dest.exists() and (expect is None or dest.stat().st_size == expect):
        print(f"  skip   {dest.name}  ({dest.stat().st_size} B)")
        return True
    url = f"{ENDPOINT}/datasets/{repo}/resolve/main/{path}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    # 镜像支持 Range（实测 206 + Content-Range），所以半截文件要留着续传：
    # 大文件（SLAKE/imgs.zip 202MB）限速时从头重下要几小时，续传只要几分钟。
    # 万一远端文件变了，续出来的大小对不上，下面的 expect 校验会拦住。
    if tmp.exists() and expect is not None and tmp.stat().st_size > expect:
        tmp.unlink()
    # 用 curl 下载：镜像的 CDN 偶尔会长时间不发数据，urllib 的逐块读会卡死，
    # curl 的 --speed-limit 能把这种连接判死并重试。
    cmd = [
        # 必须用 HTTP/1.1：镜像的 HTTP/2 多路复用会中途停住不发数据。
        "curl", "-sSL", "--http1.1", "--fail",
        "-C", "-",  # 断点续传
        "--retry", str(retries), "--retry-delay", "3", "--retry-all-errors",
        "--connect-timeout", "20", "--max-time", "1800",
        "--speed-limit", "1024", "--speed-time", "120",
        "-o", str(tmp), url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  fail   {path}: curl {proc.returncode} {proc.stderr.strip()[:160]}", file=sys.stderr)
        # 半截文件保留，下次跑接着传
        return False
    if expect is not None and tmp.stat().st_size != expect:
        print(f"  fail   {path}: size {tmp.stat().st_size} != {expect}", file=sys.stderr)
        return False
    tmp.replace(dest)
    print(f"  ok     {dest.name}  ({dest.stat().st_size} B)")
    return True


def main() -> int:
    # --exclude 用来跳过镜像限速严重的大文件（例如 SLAKE/imgs.zip），
    # 先把其余文件拉齐，大文件单独用不限时的 curl 慢慢挂；脚本本身是幂等的。
    import argparse

    parser = argparse.ArgumentParser(description="拉取测评集原始文件到 _raw/")
    parser.add_argument(
        "--exclude",
        default="",
        help="逗号分隔的子串，路径命中则跳过（例如 imgs.zip,PathVQA）",
    )
    args = parser.parse_args()
    excludes = [item.strip() for item in args.exclude.split(",") if item.strip()]
    wanted = [item for item in FILES if not any(pat in item[2] for pat in excludes)]

    RAW.mkdir(parents=True, exist_ok=True)
    failed = []
    by_repo: dict[str, list[tuple[str, str]]] = {}
    for sub, repo, path in wanted:
        by_repo.setdefault(repo, []).append((sub, path))

    sizes: dict[str, int | None] = {}
    for repo in by_repo:
        tree = repo_tree(repo)
        for _, path in by_repo[repo]:
            sizes[f"{repo}::{path}"] = tree.get(path)
        print(f"  清单 {repo}: {len(tree)} 个文件")

    for sub, repo, path in wanted:
        # 保留仓库内的相对路径：MMLU、MedQA 各子集的文件名一模一样，只用文件名会互相覆盖。
        dest = RAW / sub / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        expect = sizes.get(f"{repo}::{path}")
        legacy = RAW / sub / Path(path).name
        # 早期版本按文件名存放，大小对得上就直接搬过来，省一次重复下载。
        if not dest.exists() and legacy.exists() and legacy != dest:
            if expect is None or legacy.stat().st_size == expect:
                dest.parent.mkdir(parents=True, exist_ok=True)
                legacy.replace(dest)
        if not download(repo, path, dest, expect):
            failed.append(f"{repo}/{path}")

    if failed:
        print("\n失败：" + "\n  ".join(failed), file=sys.stderr)
        return 1
    total = sum(f.stat().st_size for f in RAW.rglob("*") if f.is_file())
    print(f"\n全部就绪，_raw/ 共 {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
