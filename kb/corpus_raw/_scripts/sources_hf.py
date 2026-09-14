#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""登记模型社区（HuggingFace / hf-mirror）上的医学语料。

策略：

1. 中文优先：考试题、临床病例、真实问诊对话、医疗百科与教材、推理数据。
2. 权威公开优先：只收许可明确（Apache-2.0 / MIT / CC-BY / CC-BY-SA）的仓库。
3. 少量无许可声明但研究常用的语料单独放，并在 license 字段写清"仅内部研究"，
   对外发布前必须再确认。

运行：python _scripts/sources_hf.py
"""

from __future__ import annotations

from common import add_sources, entry, hf_tree, hf_url, setup_stdout

# (子目录, 仓库, 许可, 说明, 仓库内文件列表)
HF_REPOS: list[tuple[str, str, str, str, list[str]]] = [
    # ---------- 中文 · 考试与临床病例 ----------
    (
        "hf_datasets/CMB",
        "FreedomIntelligence/CMB",
        "Apache-2.0",
        "中文医学考试（CMB-Exam）+ 临床病例（CMB-Clin），权威中文基准",
        [
            "CMB-Clin/CMB-Clin-qa.json",
            "CMB-Exam/CMB-val/CMB-val-merge.json",
            "CMB-Exam/CMB-test/CMB-test-choice-question-merge.json",
            "CMB-Exam/CMB-train/CMB-train-merge.json",
        ],
    ),
    (
        "hf_datasets/CMExam",
        "fzkuji/CMExam",
        "Apache-2.0",
        "中文医学考试题（含解析与知识点），中山大学发布",
        ["train.json", "valid.json", "test.json"],
    ),
    (
        "hf_datasets/PromptCBLUE",
        "tchenglv/PromptCBLUE",
        "MIT",
        "中文医疗 NLP 多任务集，含 IMCS-V2 真实问诊对话（D06 同源）",
        ["dev.json"],
    ),
    # ---------- 中文 · 对话、问答、百科、教材 ----------
    (
        "hf_datasets/Chinese-medical-dialogue",
        "ticoAg/Chinese-medical-dialogue",
        "Apache-2.0（原始内容来自公开医患问答）",
        "中文医患对话，按科室划分，适合做口语化问诊语料",
        ["data/train_0001_of_0001.json"],
    ),
    (
        "hf_datasets/Chinese-medical-QA",
        "whalning/Chinese-medical-QA",
        "MIT",
        "中文医学问答对",
        ["dataset.json"],
    ),
    (
        "hf_datasets/DISC-Med-SFT",
        "Flmc/DISC-Med-SFT",
        "Apache-2.0",
        "中文医疗对话微调数据，含患者口语表达与追问",
        ["DISC-Med-SFT_released.jsonl"],
    ),
    (
        "hf_datasets/medical-shibing624",
        "shibing624/medical",
        "Apache-2.0",
        "中文医疗综合语料：医学教材、医疗百科、对话指令数据",
        [
            "pretrain/medical_book_zh.json",
            "pretrain/train_encyclopedia.json",
            "pretrain/valid_encyclopedia.json",
            "pretrain/test_encyclopedia.json",
            "finetune/train_zh_0.json",
            "finetune/valid_en_1.json",
        ],
    ),
    (
        "hf_datasets/Medical-R1-Distill-Chinese",
        "FreedomIntelligence/Medical-R1-Distill-Data-Chinese",
        "Apache-2.0",
        "中文医学推理蒸馏数据（含诊断思维链），可用于构造推理型语料",
        ["medical_r1_distill_sft_Chinese.json"],
    ),
    # ---------- 指南与教材 ----------
    (
        "hf_datasets/meditron-clinical-guidelines",
        "zechen-nlp/meditron-clinical-guidelines",
        "CC-BY-4.0",
        "多语种临床指南全文（Meditron 整理，含中文指南），做指南层的核心语料",
        ["clinical_guidelines.jsonl"],
    ),
    # ---------- 英文 · 评测、安全与计算 ----------
    (
        "hf_datasets/HealthBench",
        "openai/healthbench",
        "MIT",
        "安全与沟通评分量表（D09–D11 同源），含负分条目",
        [
            "2025-05-07-06-14-12_oss_eval.jsonl",
            "hard_2025-05-08-21-00-10.jsonl",
            "consensus_2025-05-09-20-00-46.jsonl",
        ],
    ),
    (
        "hf_datasets/MedCalc-Bench",
        "ncbi/MedCalc-Bench",
        "CC-BY-SA-4.0",
        "临床计算题库（D12 同源），公式与参考答案",
        ["test_data_11_18_final.csv", "train_data_11_18_final.csv"],
    ),
    (
        "hf_datasets/PubMedQA",
        "qiaojin/PubMedQA",
        "MIT",
        "医学文献证据问答（D13 证据问答同源）",
        [
            "pqa_labeled/train-00000-of-00001.parquet",
            "pqa_unlabeled/train-00000-of-00001.parquet",
        ],
    ),
    (
        "hf_datasets/MedXpertQA",
        "TsinghuaC3I/MedXpertQA",
        "MIT",
        "专家级疑难病例推理（D08 同源）",
        ["Text/test.jsonl", "Text/dev.jsonl"],
    ),
    (
        "hf_datasets/MedMCQA",
        "openlifescienceai/medmcqa",
        "Apache-2.0",
        "英文医学考试题（印度 PG 医学入学考试）",
        ["data/validation-00000-of-00001.parquet"],
    ),
    (
        "hf_datasets/MMLU-medical",
        "cais/mmlu",
        "MIT",
        "MMLU 医学相关子集（知识底座）",
        [
            "clinical_knowledge/test-00000-of-00001.parquet",
            "medical_genetics/test-00000-of-00001.parquet",
            "professional_medicine/test-00000-of-00001.parquet",
            "college_medicine/test-00000-of-00001.parquet",
            "anatomy/test-00000-of-00001.parquet",
        ],
    ),
    (
        "hf_datasets/MedQA",
        "bigbio/med_qa",
        "未声明（原始数据见论文，仅内部研究）",
        "MedQA 医学考试题（中英双版本）",
        [
            "med_qa_zh_4options_bigbio_qa/test-00000-of-00001.parquet",
            "med_qa_en_4options_bigbio_qa/test-00000-of-00001.parquet",
        ],
    ),
    (
        "hf_datasets/NEJM-CPC",
        "katielink/nejm-medqa-diagnostic-reasoning-dataset",
        "CC-BY-4.0（病例正文版权属 NEJM）",
        "NEJM 临床病理讨论真实病例（鉴别诊断难例）",
        [
            "nejm_test/train-00000-of-00001.parquet",
            "41746_2024_1010_MOESM2_ESM.csv",
            "41746_2024_1010_MOESM4_ESM.csv",
        ],
    ),
    (
        "hf_datasets/AgentClinic",
        "katielink/agentclinic_medqa",
        "未声明（社区镜像，仅内部研究）",
        "智能体问诊场景（D07 同源），含隐藏病史",
        ["agentclinic_medqa.jsonl"],
    ),
    (
        "hf_datasets/CmedqaRetrieval",
        "C-MTEB/CmedqaRetrieval",
        "未声明（仅内部评测）",
        "中文医学检索评测集（D13 检索题同源），含语料与查询",
        [
            "data/corpus-00000-of-00001-a3949861f65a3226.parquet",
            "data/queries-00000-of-00001-daeedab899d3c839.parquet",
        ],
    ),
]

# 仓库文件太多时按规则取子集： (子目录, 仓库, 许可, 说明, 路径前缀, 最多文件数, 总大小上限MB)
HF_SUBSETS: list[tuple[str, str, str, str, str, int, int]] = [
    (
        "hf_datasets/MedRAG-textbooks",
        "MedRAG/textbooks",
        "未声明（教材原文版权归出版社，仅内部研究）",
        "医学教材分块语料（Harrison 内科学、Schwartz 外科学等）",
        "chunk/",
        20,
        400,
    ),
    (
        "hf_datasets/MedRAG-pubmed",
        "MedRAG/pubmed",
        "未声明（PubMed 摘要，仅内部研究）",
        "PubMed 摘要分块语料（取部分分片，非全量 55GB）",
        "chunk/",
        6,
        600,
    ),
]


def main() -> int:
    setup_stdout()
    rows: list[dict] = []

    for local_dir, repo, license_, note, files in HF_REPOS:
        for path in files:
            rows.append(
                entry(
                    category="hf_datasets",
                    source=repo,
                    license_=license_,
                    url=hf_url(repo, path),
                    local_path=f"{local_dir}/{path.split('/')[-1]}",
                    note=note,
                )
            )

    for local_dir, repo, license_, note, prefix, max_files, max_mb in HF_SUBSETS:
        try:
            tree = hf_tree(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 取清单失败 {repo}: {exc}")
            continue
        picked, total = [], 0
        for path in sorted(tree):
            if not path.startswith(prefix):
                continue
            size = tree[path]
            if total + size > max_mb * 1024 * 1024:
                break
            picked.append(path)
            total += size
            if len(picked) >= max_files:
                break
        for path in picked:
            rows.append(
                entry(
                    category="hf_datasets",
                    source=repo,
                    license_=license_,
                    url=hf_url(repo, path),
                    local_path=f"{local_dir}/{path.split('/')[-1]}",
                    note=note,
                )
            )
        print(f"[info] {repo}: 选中 {len(picked)} 个文件，{total / 1e6:.1f} MB")

    added = add_sources(rows)
    print(f"[ok] 登记 {len(rows)} 条，新增 {added} 条 -> sources.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
