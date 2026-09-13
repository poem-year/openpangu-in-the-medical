# -*- coding: utf-8 -*-
"""从公开权威中文医疗测评集中抽样，组装面向医疗诊断场景的测评集。"""
import json, io, os, re, urllib.request
from collections import Counter, OrderedDict

ROOT = r"E:\面向医疗诊断场景的openpangu能力增强方案\评测集"
RAW = os.path.join(ROOT, "_raw")

SOURCES = {
    "CMB-Exam": {
        "name": "CMB (Chinese Medical Benchmark) - CMB-Exam",
        "org": "FreedomIntelligence (上海交大 / 华东师大)",
        "url_modelscope": "https://modelscope.cn/datasets/FreedomIntelligence/CMB",
        "url_hf": "https://huggingface.co/datasets/FreedomIntelligence/CMB",
        "license": "Apache-2.0",
        "paper": "arXiv:2308.08833",
    },
    "CMB-Clin": {
        "name": "CMB - CMB-Clin (临床病例分析)",
        "org": "FreedomIntelligence",
        "url_modelscope": "https://modelscope.cn/datasets/FreedomIntelligence/CMB",
        "url_hf": "https://huggingface.co/datasets/FreedomIntelligence/CMB",
        "license": "Apache-2.0",
        "paper": "arXiv:2308.08833",
    },
    "CMExam": {
        "name": "CMExam (中国国家执业医师资格考试题)",
        "org": "中山大学等",
        "url_hf": "https://huggingface.co/datasets/fzkuji/CMExam",
        "license": "Apache-2.0",
        "paper": "arXiv:2306.03030",
    },
    "IMCS-V2": {
        "name": "IMCS-V2 (PromptCBLUE / CHIP-2023, 医患多轮问诊)",
        "org": "CBLUE 评测 / CHIP 会议",
        "url_hf": "https://huggingface.co/datasets/tchenglv/PromptCBLUE",
        "license": "MIT (PromptCBLUE 封装)",
        "paper": "IMCS-V2, CBLUE benchmark",
    },
    "cMedQA-V2.0": {
        "name": "cMedQA-V2.0 (中文医学问答)",
        "org": "wangrongsheng 镜像 / 原始 cMedQA2",
        "url_hf": "https://huggingface.co/datasets/wangrongsheng/cMedQA-V2.0",
        "license": "未标注（研究用途）",
        "paper": "cMedQA2, IEEE Access 2018",
    },
}

review = io.StringIO()

# ---------- 1. CMB-Exam val ----------
val = json.load(open(os.path.join(RAW, "CMB", "CMB-Exam", "CMB-val", "CMB-val-merge.json"), encoding="utf-8"))
print("CMB val total:", len(val), file=review)

EXCLUDE_CLASS = {"考研政治"}
CLIN_WEIGHT = {"临床医学": 3, "执业医师": 3, "执业助理": 3, "西医综合": 3, "规培结业": 3,
               "中级职称": 2, "高级职称": 2, "医学考研": 2, "护理学": 2, "医技": 2,
               "基础医学": 1, "预防医学与公共卫生学": 1}
DIAG_KW = re.compile(r"最可能|诊断|首先考虑|应考虑|应首先|鉴别|首选|典型|表现|见于|提示|确诊|治疗|处理")

pool = []
for it in val:
    if it.get("exam_class") in EXCLUDE_CLASS:
        continue
    if re.search(r"药师|中药|中医", it.get("exam_subject", "")) or re.search(r"中药|中医", it.get("exam_class", "")):
        continue
    w = CLIN_WEIGHT.get(it.get("exam_class"), 1)
    score = w * 2 + (2 if DIAG_KW.search(it.get("question", "")) else 0)
    pool.append((score, it))
pool.sort(key=lambda x: -x[0])
print("CMB val pool after filter:", len(pool), file=review)
print("pool by class:", Counter(x[1]["exam_class"] for x in pool), file=review)

# 分层抽样：每个 exam_class 最多 6 题，优先高分
sel_a, per_class = [], Counter()
for score, it in pool:
    c = it.get("exam_class")
    if per_class[c] >= 6:
        continue
    sel_a.append(it)
    per_class[c] += 1
    if len(sel_a) >= 50:
        break
print("CMB val selected:", len(sel_a), "| by class:", per_class, file=review)
print(file=review)
print("=== CMB-Exam 选中题目 ===", file=review)
for i, it in enumerate(sel_a, 1):
    print(f"{i:>2}. [{it['exam_class']}/{it['exam_subject']}] {it['question'][:66]} => {it['answer']}", file=review)

open(os.path.join(RAW, "_review.txt"), "w", encoding="utf-8").write(review.getvalue())
print("review written")