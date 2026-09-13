#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建正式诊断测评集用的公共定义：来源档案、统一字段、写出工具。"""

import json
import re
import unicodedata
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "_raw"

COLLECTED_AT = date.today().isoformat()

# 每个来源的档案信息。licence 以 HuggingFace 仓库卡片的 license 字段为准，
# 卡片未声明的一律标「未声明」，对外材料里不要写成已授权。
SRC = {
    "CMB": {
        "name": "CMB（含 CMB-Exam / CMB-Clin）",
        "org": "上海交通大学 等",
        "url": "https://huggingface.co/datasets/FreedomIntelligence/CMB",
        "license": "Apache-2.0",
        "paper": "arXiv:2308.08833",
        "published": "2023-07",
        "updated": "2024-04-05",
        "contamination_risk": "高",
        "note": "中文医疗基准；CMB-Clin 为真实住院病历级病例，公开数据中少见。",
    },
    "CMExam": {
        "name": "CMExam",
        "org": "中山大学 等",
        "url": "https://huggingface.co/datasets/fzkuji/CMExam",
        "license": "Apache-2.0",
        "paper": "arXiv:2306.03030",
        "published": "2023-06",
        "updated": "2024-11-02",
        "contamination_risk": "高",
        "note": "中国国家执业医师资格考试题，含官方解析。",
    },
    "PromptCBLUE": {
        "name": "PromptCBLUE（含 IMCS-V2 等子集）",
        "org": "CHIP-2023 中文医疗信息处理评测",
        "url": "https://huggingface.co/datasets/tchenglv/PromptCBLUE",
        "license": "MIT",
        "paper": "arXiv:2310.14151",
        "published": "2023-10",
        "updated": "2025-05-09",
        "contamination_risk": "中",
        "note": "中文医疗 NLP 多任务合集，含真实医患多轮问诊。",
    },
    "MMLU": {
        "name": "MMLU（医学相关子集）",
        "org": "UC Berkeley 等",
        "url": "https://huggingface.co/datasets/cais/mmlu",
        "license": "MIT",
        "paper": "arXiv:2009.03300",
        "published": "2020-09",
        "updated": "2024-03-08",
        "contamination_risk": "极高",
        "note": "通用知识基准的医学子集，只作知识底座用，不代表诊断能力。",
    },
    "MedQA": {
        "name": "MedQA（USMLE，中英双版本）",
        "org": "BigBio 整理，原始数据 Jin et al.",
        "url": "https://huggingface.co/datasets/bigbio/med_qa",
        "license": "仓库未声明（原始数据见论文）",
        "paper": "arXiv:2009.13081",
        "published": "2020-09",
        "updated": "2026-09-04",
        "contamination_risk": "极高",
        "note": "美国执业医师考试题；本集用其中文版为主、英文版为辅。",
    },
    "MedMCQA": {
        "name": "MedMCQA",
        "org": "IIT Kharagpur 等",
        "url": "https://huggingface.co/datasets/openlifescienceai/medmcqa",
        "license": "Apache-2.0",
        "paper": "arXiv:2203.14371",
        "published": "2022-03",
        "updated": "2024-01-04",
        "contamination_risk": "极高",
        "note": "印度医学入学/执业考试题。",
    },
    "NEJM_CPC": {
        "name": "NEJM 临床病理讨论（Diagnostic Reasoning）",
        "org": "NEJM CPC 病例，katielink 整理",
        "url": "https://huggingface.co/datasets/katielink/nejm-medqa-diagnostic-reasoning-dataset",
        "license": "CC-BY-4.0（病例正文版权属 NEJM）",
        "paper": "npj Digital Medicine 2024, s41746-024-01010-1",
        "published": "2024-01",
        "updated": "2024-01-25",
        "contamination_risk": "低",
        "note": "真实疑难病例（CPC），金标准为最终病理诊断；对外使用时注意病例版权。",
    },
    "MedXpertQA": {
        "name": "MedXpertQA",
        "org": "清华大学 等",
        "url": "https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA",
        "license": "MIT",
        "paper": "arXiv:2501.18362",
        "published": "2025-01",
        "updated": "2025-07-09",
        "contamination_risk": "低",
        "note": "2025 年发布的高难度专家级医学推理集，有文本与多模态两部分。",
    },
    "AgentClinic": {
        "name": "AgentClinic（MedQA 场景版）",
        "org": "AIM-Harvard，katielink 镜像",
        "url": "https://huggingface.co/datasets/katielink/agentclinic_medqa",
        "license": "未声明（社区镜像）",
        "paper": "arXiv:2405.07960",
        "published": "2024-05",
        "updated": "2024-06-20",
        "contamination_risk": "中",
        "note": "智能体式问诊场景：模型需主动采集信息才能给出诊断。",
    },
    "HealthBench": {
        "name": "HealthBench",
        "org": "OpenAI",
        "url": "https://huggingface.co/datasets/openai/healthbench",
        "license": "MIT",
        "paper": "arXiv:2505.08775",
        "published": "2025-05",
        "updated": "2025-08-27",
        "contamination_risk": "低",
        "note": "真实临床场景 + 医生撰写的评分量表（rubric），覆盖急症、沟通、安全。",
    },
    "MedCalcBench": {
        "name": "MedCalc-Bench",
        "org": "NIH / NCBI 等",
        "url": "https://huggingface.co/datasets/ncbi/MedCalc-Bench",
        "license": "CC-BY-SA-4.0",
        "paper": "arXiv:2406.12036",
        "published": "2024-06",
        "updated": "2025-12-18",
        "contamination_risk": "低",
        "note": "临床计算题：由病历算出评分/指标，考工具调用与数值计算。",
    },
    "PubMedQA": {
        "name": "PubMedQA",
        "org": "韩国科学技术院 等",
        "url": "https://huggingface.co/datasets/qiaojin/PubMedQA",
        "license": "MIT",
        "paper": "arXiv:1909.06146",
        "published": "2019-09",
        "updated": "2024-03-06",
        "contamination_risk": "中",
        "note": "给定文献摘要回答研究结论，考证据使用与不越界推断。",
    },
    "CmedqaRetrieval": {
        "name": "CmedqaRetrieval（C-MTEB）",
        "org": "C-MTEB / 中文医疗问答语料",
        "url": "https://huggingface.co/datasets/C-MTEB/CmedqaRetrieval",
        "license": "未声明（C-MTEB 汇总集）",
        "paper": "arXiv:2309.07597",
        "published": "2023-07",
        "updated": "2023-07-28",
        "contamination_risk": "中",
        "note": "中文医学检索评测：10 万篇语料 + 3999 条查询，测检索召回。",
    },
}

LEVELS = {
    "D01": ["L0", "L1"],
    "D02": ["L0", "L1", "L2", "L3"],
    "D03": ["L1", "L2", "L3", "L4"],
    "D04": ["L1", "L2", "L4"],
    "D05": ["L0", "L1", "L2", "L4"],
    "D06": ["L1", "L2", "L4"],
    "D07": ["L4", "L5"],
    "D08": ["L1", "L3", "L4"],
    "D09": ["L4", "L5"],
    "D10": ["L4", "L5"],
    "D11": ["L4", "L5"],
    "D12": ["L4", "L5"],
    "D13": ["L2", "L3"],
}


# 参考答案里常用 ①②③ 分点。NFKC 会把它们压成 123，丢掉分点结构，
# 归一化前后临时替换掉，保证正文保真。
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"


def clean(text) -> str:
    """去掉多余空白与不可见字符，不动正文内容。"""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    for i, ch in enumerate(_CIRCLED):
        text = text.replace(ch, f"\x00{i}\x00")
    text = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    for i, ch in enumerate(_CIRCLED):
        text = text.replace(f"\x00{i}\x00", ch)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_zh(text: str) -> bool:
    if not text:
        return False
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return han / max(len(text), 1) > 0.15


def item(dim_code: str, dim_name: str, seq: int, *, task_type: str, question: str,
         src_key: str, scoring: dict, difficulty: str = "medium",
         options=None, answer=None, reference_answer=None, key_points=None,
         rubric=None, context=None, dialogue=None, media=None,
         extra_source=None, capability=None, notes=None) -> dict:
    """按统一字段造一道题。所有维度共用这一套字段，方便下游一套评测程序跑全库。"""
    q = clean(question)
    src = dict(SRC[src_key])
    if extra_source:
        src["raw_file"] = extra_source
    out = {
        "id": f"{dim_code}-{seq:04d}",
        "dimension_code": dim_code,
        "dimension": dim_name,
        "capability": capability,
        "task_type": task_type,
        # 按「提问 + 病例正文」一起判定语种。有些维度的提问是中文模板，
        # 但真正的题目内容（如 NEJM 病例）是英文，只看题干会判错。
        "language": "zh" if is_zh(q + " " + clean(context or "")) else "en",
        "question": q,
        "options": options,
        "answer": answer,
        "reference_answer": reference_answer,
        "key_points": key_points,
        "rubric": rubric,
        "context": clean(context) or None,
        "dialogue": dialogue,
        "media": media,
        "scoring": scoring,
        "level_hint": LEVELS[dim_code],
        "difficulty": difficulty,
        "source": src,
        "provenance": {
            "collected_at": COLLECTED_AT,
            "dataset_published": src["published"],
            "dataset_updated": src["updated"],
            "contamination_risk": src["contamination_risk"],
        },
    }
    if notes:
        out["notes"] = notes
    return out


# 需要翻译的字段。机器翻译只碰这些内容字段，不碰 id、指标名、来源等结构化字段。
TRANSLATABLE_KEYS = {"question", "reference_answer", "context", "options",
                     "key_points", "rubric", "dialogue", "criterion", "text"}
# 结构性字段：即使嵌在可翻译块里也保持原样（role 是程序用的，axis/points 是判分用的）
SKIP_KEYS = {"role", "axis", "points", "id"}


def iter_strings(obj, path=(), inside=False):
    """递归取出可翻译的字符串，返回 (路径, 文本)。

    inside 表示已经进入某个可翻译块（如 options 的 A/B/C 键），
    此时内部的键名不做白名单判断，否则 {"A": "..."} 这种结构会被漏掉。
    """
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_strings(v, path + (i,), inside)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            if inside or k in TRANSLATABLE_KEYS:
                yield from iter_strings(v, path + (k,), True)


def set_path(obj, path, value):
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value


def get_path(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def write_items(items: list[dict], filename: str) -> Path:
    # 统一按顺序重新编号，保证 id 形如 D03-0007 且每个维度内唯一、连续。
    for i, it in enumerate(items, start=1):
        it["id"] = f"{it['dimension_code']}-{i:04d}"
    path = ROOT / filename
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  写出 {filename}: {len(items)} 题")
    return path


def load_json(name: str):
    with (RAW / name).open(encoding="utf-8") as fh:
        return json.load(fh)
