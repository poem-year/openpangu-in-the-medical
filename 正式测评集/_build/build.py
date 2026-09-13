#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 _raw/ 里的权威公开集抽取，产出 13 个维度的 jsonl。

原则：
1. 不自编题目，所有题目来自公开测评集，只做筛选、抽样、格式统一；
2. 以中文为主。英文来源的题目用机器翻译转成中文，仅保留少量英文原题，
   译前原文存在 original_en 字段里，便于逐条核对；
3. 每个维度对应一种可单独解释的能力，治疗、护理、中医理论等题目剔除；
4. 抽样固定随机种子 + 等距抽样，保证可复现、不挑题。
"""

import ast
import collections
import json
import random
import re
from pathlib import Path

import pandas as pd

from common import (RAW, ROOT, clean, get_path, is_zh, item, iter_strings,
                    load_json, set_path, write_items)
from translate import Translator

SEED = 20260913
MEDIA = ROOT / "media"


def spread(seq: list, k: int) -> list:
    """等距抽样：在整段序列上均匀取 k 个，避免只取头部带来的偏置。"""
    n = len(seq)
    if n <= k:
        return list(seq)
    idx, seen = [], set()
    for i in range(k):
        j = round(i * (n - 1) / (k - 1))
        if j not in seen:
            seen.add(j)
            idx.append(j)
    return [seq[j] for j in idx]


def as_list(value):
    """HF 的 choices 列有时是 ndarray，有时是 repr 字符串。"""
    if isinstance(value, str):
        try:
            return list(ast.literal_eval(value))
        except Exception:
            return [value]
    return list(value)


def dedup(rows: list, key=None) -> list:
    """按题干去重，保留先出现的。源数据里存在重复题，抽样前先去掉，
    否则会挤掉本来可以选进来的其他题。"""
    seen, out = set(), []
    for r in rows:
        k = clean(key(r) if key else r["question"])[:150]
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def match_answer(options: list, answer_text) -> str | None:
    """MedQA 的 answer 存的是选项原文，不是字母，这里映射回字母。"""
    target = clean(answer_text)
    for i, opt in enumerate(options):
        if clean(opt) == target:
            return chr(65 + i)
    return None


CASE_RE = re.compile(r"患者|病人|男，|女，|男性|女性|\d+\s*岁|岁男|岁女")
EN_CASE_RE = re.compile(r"year[-\s]old|presented with|patient|man|woman|boy|girl", re.I)
DX_ASK_RE = re.compile(r"最可能|应诊断为|诊断为|首先考虑|考虑为|诊断是|初步诊断|最可能的诊断")
DDX_ASK_RE = re.compile(r"鉴别|需与.{0,10}(区别|区分)|相鉴别|除外|鉴别诊断")
WORKUP_RE = re.compile(r"首选.{0,6}检查|最有价值.{0,6}检查|最合适.{0,6}检查|明确诊断.{0,8}检查|"
                       r"确诊.{0,6}检查|进一步.{0,4}检查|应做.{0,6}检查|首选的检查|最有意义的检查")
WORKUP_ASK_RE = re.compile(
    r"(哪些|什么|何种|需|应|要).{0,8}检查|检查.{0,12}(明确诊断|确诊|进一步|优先)"
    r"|(明确诊断|确诊).{0,10}检查|首选.{0,6}检查|最有价值.{0,6}检查")
EXCLUDE_RE = re.compile(r"治疗|用药|药物|护理|手术|预防接种|药理|中医|脉象|方剂|伦理|统计|"
                        r"解剖结构|药代|护理措施|健康宣教|康复训练")


def ask_stem(q: str, n: int = 100) -> str:
    """只取问句落点来判断这题在问什么。

    病例正文里出现「治疗史」「既往用药」是正常的，不能因此把整题判成治疗题；
    真正决定题目类型的是最后那句问句。
    """
    return q[-n:]


def _split_points(text: str) -> list:
    """把参考答案拆成要点，供覆盖度打分。"""
    t = clean(text)
    parts = re.split(r"[①②③④⑤⑥⑦⑧⑨⑩]|\n\s*\d+[\.、]|；|\n(?=\S)", t)
    out = [clean(p) for p in parts]
    return [p for p in out if len(p) >= 6][:12]


def _cmexam_rows(name: str):
    for line in (RAW / "CMExam" / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _medqa_zh():
    return pd.read_parquet(RAW / "MedQA" / "med_qa_zh_4options_bigbio_qa" / "test-00000-of-00001.parquet")


# ---------------------------------------------------------------- D01 知识底座

MMLU_SUBJECTS = [
    ("clinical_knowledge", "临床知识"),
    ("professional_medicine", "专业医学"),
    ("college_medicine", "医学基础"),
]


def build_d01():
    zh = []
    for r in load_json("CMB/CMB-Exam/CMB-val/CMB-val-merge.json"):
        if r.get("exam_class") not in {"临床医学", "西医综合", "执业医师", "执业助理医师",
                                       "中级职称", "规培结业", "基础医学", "医学考研"}:
            continue
        ans = clean(r.get("answer"))
        opt = r.get("option") or {}
        if len(ans) != 1 or ans not in opt:      # 只取单选，多选判分口径不一样
            continue
        zh.append(item(
            "D01", "医学知识基础", 0, task_type="mcq_single",
            question=r["question"], src_key="CMB", options=opt, answer=ans,
            reference_answer=clean(r.get("explanation")) or None,
            scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
            capability="中文医学考试知识点", difficulty="easy",
            extra_source="CMB-val-merge.json",
            notes=f"考试类别：{r.get('exam_class')}／{r.get('exam_subject')}",
        ))
    cmb = spread(dedup(zh), 25)

    # CMExam 里挑没有病例正文的知识型题，和 D02 的病例题互斥
    ex = []
    for r in _cmexam_rows("test.json"):
        q = clean(r["Question"])
        if CASE_RE.search(q):
            continue
        opt = {o["key"]: clean(o["value"]) for o in r["Options"]}
        ans = clean(r["Answer"])
        if ans not in opt:
            continue
        ex.append(item(
            "D01", "医学知识基础", 0, task_type="mcq_single", question=q,
            src_key="CMExam", options=opt, answer=ans,
            reference_answer=clean(r.get("Explanation")) or None,
            scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
            capability="执业医师考试知识点", difficulty="easy",
            extra_source="CMExam test.json",
        ))
    cmexam = spread(dedup(ex), 10)

    # 保留少量英文题，用于看跨语言一致性
    en = []
    for slug, cn in MMLU_SUBJECTS:
        df = pd.read_parquet(RAW / "MMLU" / slug / "test-00000-of-00001.parquet")
        for _, r in df.iterrows():
            opts = as_list(r["choices"])
            opt = {chr(65 + i): clean(o) for i, o in enumerate(opts)}
            en.append(item(
                "D01", "医学知识基础", 0, task_type="mcq_single",
                question=r["question"], src_key="MMLU", options=opt,
                answer=chr(65 + int(r["answer"])),
                scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
                capability="英文医学知识点（跨语言对照）", difficulty="easy",
                extra_source=f"MMLU/{slug}/test", notes=f"MMLU 子集：{cn}",
            ))
    return cmb + cmexam + spread(en, 5)


# ------------------------------------------------------------ D02 病例诊断推理

def build_d02():
    cand = []
    for _, r in _medqa_zh().iterrows():
        q = clean(r["question"])
        tail = ask_stem(q)
        if not CASE_RE.search(q) or not DX_ASK_RE.search(tail) or EXCLUDE_RE.search(tail):
            continue
        opts = as_list(r["choices"])
        opt = {chr(65 + i): clean(o) for i, o in enumerate(opts)}
        ans = match_answer(opts, as_list(r["answer"])[0])
        if not ans:
            continue
        cand.append(item(
            "D02", "病例诊断推理", 0, task_type="mcq_single", question=q,
            src_key="MedQA", options=opt, answer=ans,
            context=clean(r.get("context")) or None,
            scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
            capability="从病例摘要推出最可能诊断", difficulty="medium",
            extra_source="med_qa_zh_4options/test",
        ))
    medqa = spread(dedup(cand), 23)

    ex = []
    for r in _cmexam_rows("test.json"):
        q = clean(r["Question"])
        tail = ask_stem(q)
        if not CASE_RE.search(q) or not DX_ASK_RE.search(tail) or EXCLUDE_RE.search(tail):
            continue
        if WORKUP_ASK_RE.search(tail) or DDX_ASK_RE.search(tail):
            continue
        opt = {o["key"]: clean(o["value"]) for o in r["Options"]}
        ans = clean(r["Answer"])
        if ans not in opt:
            continue
        ex.append(item(
            "D02", "病例诊断推理", 0, task_type="mcq_single", question=q,
            src_key="CMExam", options=opt, answer=ans,
            reference_answer=clean(r.get("Explanation")) or None,
            scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
            capability="执业医师考试风格的病例诊断", difficulty="medium",
            extra_source="CMExam test.json",
        ))
    cmexam = spread(dedup(ex), 15)

    den = pd.read_parquet(RAW / "MedQA" / "med_qa_en_4options_bigbio_qa" / "test-00000-of-00001.parquet")
    ecand = []
    for _, r in den.iterrows():
        q = clean(r["question"])
        if not EN_CASE_RE.search(q) or not re.search(r"most likely|diagnosis|diagnose", ask_stem(q), re.I):
            continue
        opts = as_list(r["choices"])
        opt = {chr(65 + i): clean(o) for i, o in enumerate(opts)}
        ans = match_answer(opts, as_list(r["answer"])[0])
        if not ans:
            continue
        ecand.append(item(
            "D02", "病例诊断推理", 0, task_type="mcq_single", question=q,
            src_key="MedQA", options=opt, answer=ans,
            scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
            capability="英文病例诊断（跨语言对照）", difficulty="medium",
            extra_source="med_qa_en_4options/test",
        ))
    return medqa + cmexam + spread(dedup(ecand), 2)


# ---------------------------------------------------------------- D03 鉴别诊断

def build_d03():
    out = []

    # 中文鉴别题：住院病历小问 + 考试题里的鉴别题
    for case in load_json("CMB/CMB-Clin/CMB-Clin-qa.json"):
        for qa in case["QA_pairs"]:
            q = clean(qa["question"])
            if not q or EXCLUDE_RE.search(q) or "鉴别" not in q or "诊断" in q:
                continue
            out.append(item(
                "D03", "鉴别诊断", 0, task_type="open_ddx",
                question=q, src_key="CMB", context=case["description"],
                reference_answer=clean(qa["answer"]),
                key_points=_split_points(qa["answer"]),
                scoring={"method": "应鉴别疾病的召回率 + 人工评分",
                         "metrics": ["ddx_recall", "rationale_quality"]},
                capability="列出需要鉴别的疾病，避免漏诊", difficulty="hard",
                extra_source="CMB-Clin-qa.json",
                notes=f"来源病例：{case.get('title')}",
            ))
    clin = spread(dedup(out), 7)

    mcq = []
    for r in _cmexam_rows("test.json"):
        q = clean(r["Question"])
        tail = ask_stem(q)
        if not CASE_RE.search(q) or EXCLUDE_RE.search(tail) or not DDX_ASK_RE.search(tail):
            continue
        opt = {o["key"]: clean(o["value"]) for o in r["Options"]}
        ans = clean(r["Answer"])
        if ans not in opt:
            continue
        mcq.append(item(
            "D03", "鉴别诊断", 0, task_type="mcq_single", question=q,
            src_key="CMExam", options=opt, answer=ans,
            reference_answer=clean(r.get("Explanation")) or None,
            scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
            capability="考试风格的鉴别诊断", difficulty="medium",
            extra_source="CMExam test.json",
        ))
    for _, r in _medqa_zh().iterrows():
        q = clean(r["question"])
        tail = ask_stem(q)
        if not CASE_RE.search(q) or EXCLUDE_RE.search(tail) or not DDX_ASK_RE.search(tail):
            continue
        opts = as_list(r["choices"])
        opt = {chr(65 + i): clean(o) for i, o in enumerate(opts)}
        ans = match_answer(opts, as_list(r["answer"])[0])
        if not ans:
            continue
        mcq.append(item(
            "D03", "鉴别诊断", 0, task_type="mcq_single", question=q,
            src_key="MedQA", options=opt, answer=ans,
            scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
            capability="考试风格的鉴别诊断", difficulty="medium",
            extra_source="med_qa_zh_4options/test",
        ))
    mcq_zh = spread(dedup(mcq), 13)

    # 保留少量英文真实疑难病例，走 top-k 命中口径
    df = pd.read_csv(RAW / "NEJM_CPC" / "41746_2024_1010_MOESM2_ESM.csv", encoding="latin-1")
    seen, cand = set(), []
    for _, r in df.iterrows():
        case = clean(r.get("question"))
        dx = clean(str(r.get("answer")).replace('"', ""))
        if len(case) < 400 or not dx or dx.lower() in {"nan", "none"}:
            continue
        if case[:120] in seen:
            continue
        seen.add(case[:120])
        cand.append((case, dx))
    en = []
    for case, dx in spread(cand, 5):
        en.append(item(
            "D03", "鉴别诊断", 0, task_type="open_ranked_ddx",
            question="请阅读该病例，给出按可能性从高到低排序的鉴别诊断列表（建议 5～10 个）。",
            src_key="NEJM_CPC", context=case, reference_answer=dx,
            scoring={
                "method": "命中率评分：最终诊断是否出现在模型给出的前 k 个鉴别诊断中",
                "metrics": ["ddx_top5_hit", "ddx_top10_hit", "mrr"],
                "protocol": "沿用 npj Digital Medicine 2024 (s41746-024-01010-1) 的鉴别诊断评分口径："
                            "金标准为病例最终诊断，按 top-k 命中计分。",
            },
            capability="英文疑难病例的鉴别诊断排序", difficulty="hard",
            extra_source="41746_2024_1010_MOESM2_ESM.csv",
            notes="NEJM 临床病理讨论（CPC）病例；金标准为最终病理/临床诊断。",
        ))
    return clin + mcq_zh + en


# ---------------------------------------------------------------- D04 检查决策

def cmb_clin_buckets() -> dict:
    """CMB-Clin 每个病例带若干小问，同一小问不能同时算进两个维度。"""
    buckets = {"ddx": [], "workup": [], "dx": []}
    for case in load_json("CMB/CMB-Clin/CMB-Clin-qa.json"):
        for qa in case["QA_pairs"]:
            q = clean(qa["question"])
            if not q or EXCLUDE_RE.search(q):
                continue
            if "鉴别" in q:
                bucket = "dx" if "诊断" in q else "ddx"
            elif WORKUP_ASK_RE.search(q):
                bucket = "workup"
            elif "诊断" in q:
                bucket = "dx"
            else:
                continue
            buckets[bucket].append((case, qa))
    return buckets


def build_d04():
    out = []
    for case, qa in cmb_clin_buckets()["workup"]:
        out.append(item(
            "D04", "检查决策", 0, task_type="open_workup",
            question=clean(qa["question"]), src_key="CMB", context=case["description"],
            reference_answer=clean(qa["answer"]),
            key_points=_split_points(qa["answer"]),
            scoring={"method": "要点覆盖 + 人工评分",
                     "metrics": ["key_point_recall", "over_testing_rate"]},
            capability="为明确诊断选择恰当的检查，避免过度检查",
            difficulty="medium", extra_source="CMB-Clin-qa.json",
            notes=f"来源病例：{case.get('title')}",
        ))
    mcq = []
    for r in _cmexam_rows("test.json"):
        q = clean(r["Question"])
        tail = ask_stem(q)
        if not CASE_RE.search(q) or EXCLUDE_RE.search(tail) or not WORKUP_RE.search(tail):
            continue
        opt = {o["key"]: clean(o["value"]) for o in r["Options"]}
        ans = clean(r["Answer"])
        if ans not in opt:
            continue
        mcq.append(item(
            "D04", "检查决策", 0, task_type="mcq_single", question=q,
            src_key="CMExam", options=opt, answer=ans,
            reference_answer=clean(r.get("Explanation")) or None,
            scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
            capability="执业医师风格的检查选择", difficulty="medium",
            extra_source="CMExam test.json",
        ))
    for _, r in _medqa_zh().iterrows():
        q = clean(r["question"])
        tail = ask_stem(q)
        if not CASE_RE.search(q) or EXCLUDE_RE.search(tail) or not WORKUP_RE.search(tail):
            continue
        opts = as_list(r["choices"])
        opt = {chr(65 + i): clean(o) for i, o in enumerate(opts)}
        ans = match_answer(opts, as_list(r["answer"])[0])
        if not ans:
            continue
        mcq.append(item(
            "D04", "检查决策", 0, task_type="mcq_single", question=q,
            src_key="MedQA", options=opt, answer=ans,
            scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
            capability="诊断性检查的优先级判断", difficulty="medium",
            extra_source="med_qa_zh_4options/test",
        ))
    return spread(dedup(out), 12) + spread(dedup(mcq), 10)


# ------------------------------------------------------------ D05 诊断分析开放题

def build_d05():
    out = []
    for case, qa in cmb_clin_buckets()["dx"]:
        out.append(item(
            "D05", "病例诊断分析", 0, task_type="open_diagnosis",
            question=clean(qa["question"]), src_key="CMB", context=case["description"],
            reference_answer=clean(qa["answer"]),
            key_points=_split_points(qa["answer"]),
            scoring={"method": "主诊断命中 + 诊断依据要点覆盖 + 人工评分",
                     "metrics": ["accuracy", "key_point_recall", "citation_traceability"]},
            capability="写出诊断并给出支撑依据，结论可追溯",
            difficulty="medium", extra_source="CMB-Clin-qa.json",
            notes=f"来源病例：{case.get('title')}",
        ))
    return spread(dedup(out), 30)


# ------------------------------------------------------------ D06 多轮问诊诊断

DX_LINE_RE = re.compile(r"诊断[：:]\s*([^\n]+)")


def build_d06():
    rows = [json.loads(l) for l in (RAW / "PromptCBLUE" / "dev.json").read_text(encoding="utf-8").splitlines() if l.strip()]
    out, seen = [], set()
    for r in rows:
        if r.get("task_dataset") != "IMCS-V2-MRG":
            continue
        text = r["input"]
        marker = "问诊对话历史："
        if marker not in text:
            continue
        m = DX_LINE_RE.search(r.get("target") or "")
        if not m or not clean(m.group(1)):
            continue
        turns = []
        for line in clean(text.split(marker, 1)[1]).splitlines():
            line = clean(line)
            if line.startswith(("患者：", "患者:")):
                turns.append({"role": "患者", "text": line.split("：", 1)[-1]})
            elif line.startswith(("医生：", "医生:")):
                turns.append({"role": "医生", "text": line.split("：", 1)[-1]})
        if len(turns) < 6:
            continue
        dkey = "".join(t["text"] for t in turns)[:200]     # 同一对话只保留一份报告
        if dkey in seen:
            continue
        seen.add(dkey)
        out.append(item(
            "D06", "多轮问诊诊断", 0, task_type="dialogue_diagnosis",
            question="阅读以下真实医患问诊对话，给出诊断结论、鉴别方向与判断依据。",
            src_key="PromptCBLUE", dialogue=turns,
            reference_answer=clean(m.group(1)),
            scoring={"method": "诊断名匹配 + 依据要点人工评分",
                     "metrics": ["accuracy", "rationale_quality"]},
            capability="从口语化多轮问诊中抓关键信息并下诊断",
            difficulty="hard", extra_source="PromptCBLUE dev.json (IMCS-V2-MRG)",
            notes=f"对话轮数：{len(turns)}",
        ))
    return spread(out, 20)


# ------------------------------------------------------------ D07 主动问诊智能体

def build_d07():
    rows = [json.loads(l) for l in (RAW / "AgentClinic" / "agentclinic_medqa.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        o = r.get("OSCE_Examination") or {}
        dx = clean(o.get("Correct_Diagnosis"))
        actor = o.get("Patient_Actor") or {}
        sym = actor.get("Symptoms")
        opening = clean(sym.get("Primary_Symptom") if isinstance(sym, dict) else sym)
        if not dx or not actor or not opening:
            continue
        ctx = "\n".join([
            f"接诊目标：{clean(o.get('Objective_for_Doctor'))}",
            "以下信息只对模拟患者可见，医生必须主动提问才能获得：",
            f"- 人口学：{clean(actor.get('Demographics'))}",
            f"- 病史：{clean(actor.get('History'))}",
            f"- 症状：{json.dumps(sym, ensure_ascii=False) if isinstance(sym, dict) else clean(sym)}",
            f"- 既往史：{clean(actor.get('Past_Medical_History'))}",
            f"- 社会史：{clean(actor.get('Social_History'))}",
            f"- 系统回顾：{clean(actor.get('Review_of_Systems'))}",
            f"- 查体：{json.dumps(o.get('Physical_Examination_Findings'), ensure_ascii=False)}",
            f"- 检查结果：{json.dumps(o.get('Test_Results'), ensure_ascii=False)}",
        ])
        out.append(item(
            "D07", "主动问诊（智能体）", 0, task_type="agentic_history_taking",
            question=(f"你扮演接诊医生。患者主诉：{opening}。"
                      "请通过向模拟患者提问来采集信息，然后给出诊断。"
                      "评分同时看诊断是否正确、关键信息是否问到、提问是否冗余。"),
            src_key="AgentClinic", context=ctx, reference_answer=dx,
            scoring={"method": "模拟患者交互后评分",
                     "metrics": ["diagnosis_accuracy", "key_info_recall",
                                 "question_efficiency", "unsafe_advice_rate"],
                     "protocol": "标准化病人（OSCE）协议：隐藏信息只对模拟患者可见，"
                                 "医生角色需主动提问才可获得。"},
            capability="在信息不全时主动采集病史，而不是凭空猜诊断",
            difficulty="hard", extra_source="agentclinic_medqa.jsonl",
        ))
    return spread(dedup(out), 15)


# ------------------------------------------------------------ D08 疑难病例全流程

def build_d08():
    rows = [json.loads(l) for l in (RAW / "MedXpertQA" / "Text" / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cand = [r for r in rows if r.get("medical_task") == "Diagnosis"]
    out = []
    for r in spread(cand, 25):
        # MedXpertQA 把选项内嵌在题干末尾，和 options 字段重复，这里剥掉只留干净题干。
        q = re.split(r"\n?\s*Answer Choices:", clean(r["question"]), maxsplit=1)[0]
        out.append(item(
            "D08", "疑难病例全流程", 0, task_type="mcq_single",
            question=q, src_key="MedXpertQA",
            options={k: clean(v) for k, v in (r.get("options") or {}).items()},
            answer=clean(r.get("label")),
            scoring={"method": "选项精确匹配", "metrics": ["accuracy"]},
            capability="专家级高难度诊断推理，需整合多个系统与长链条证据",
            difficulty="hard", extra_source="MedXpertQA Text/test.jsonl",
            notes=f"medical_task={r.get('medical_task')}／{r.get('question_type')}／"
                  f"system={r.get('body_system')}",
        ))
    return out


# ------------------------------------------------------------ HealthBench 切分

def _healthbench_items(rows: list, dim_code: str, dim_name: str, capability: str,
                       difficulty: str = "medium") -> list:
    out = []
    for r in rows:
        msgs = r.get("prompt") or []
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        if not user_msgs:
            continue
        question = clean(user_msgs[-1]["content"])
        dialogue = None
        if len(msgs) > 1:
            dialogue = [{"role": m.get("role"), "text": clean(m.get("content"))} for m in msgs]
        rubrics = []
        for rb in r.get("rubrics") or []:
            tags = rb.get("tags") or []
            rubrics.append({
                "criterion": clean(rb.get("criterion")),
                "points": rb.get("points"),
                "axis": next((t.split(":", 1)[1] for t in tags if t.startswith("axis:")), None),
            })
        out.append(item(
            dim_code, dim_name, 0, task_type="rubric_scored",
            question=question, src_key="HealthBench", dialogue=dialogue,
            rubric=rubrics,
            scoring={"method": "按医生撰写的 rubric 逐条给分（正分加分、负分扣分）",
                     "metrics": ["rubric_score", "safe_behavior_rate", "over_refusal_rate"]},
            capability=capability, difficulty=difficulty,
            extra_source="HealthBench oss_eval.jsonl",
            notes="场景标签：" + "、".join(t for t in r.get("example_tags", []) if t.startswith("theme:")),
        ))
    return out


def _hb_rows(name: str) -> list:
    return [json.loads(l) for l in (RAW / "HealthBench" / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_d09():
    rows = _hb_rows("2025-05-07-06-14-12_oss_eval.jsonl")

    def neg_count(r):
        return sum(1 for rb in r.get("rubrics", []) if (rb.get("points") or 0) < 0)

    cand = [r for r in rows if "theme:emergency_referrals" in (r.get("example_tags") or [])
            or neg_count(r) >= 2]
    cand.sort(key=neg_count, reverse=True)
    return _healthbench_items(
        spread(cand, 25), "D09", "安全与急症边界",
        "识别急症与红线情形，正确转诊、不给危险建议", difficulty="hard")


def build_d10():
    rows = _hb_rows("2025-05-07-06-14-12_oss_eval.jsonl")
    ctx_tags = {"physician_agreed_category:not-enough-context",
                "physician_agreed_category:context-matters-but-unclear",
                "physician_agreed_category:any-reducible-uncertainty"}
    cand = [r for r in rows
            if "theme:context_seeking" in (r.get("example_tags") or [])
            or "theme:hedging" in (r.get("example_tags") or [])
            or ctx_tags & set(r.get("example_tags") or [])]
    return _healthbench_items(
        spread(cand, 20), "D10", "信息不足与不确定性",
        "信息不足时先追问而不是硬答，并对不确定性如实表达")


def build_d11():
    rows = _hb_rows("2025-05-07-06-14-12_oss_eval.jsonl")
    cand = []
    for r in rows:
        if "theme:communication" not in (r.get("example_tags") or []):
            continue
        if any("axis:communication_quality" in (rb.get("tags") or [])
               for rb in r.get("rubrics", [])):
            cand.append(r)
    return _healthbench_items(
        spread(cand, 20), "D11", "医患沟通与解释",
        "把诊断结论讲清楚、讲得患者能懂，且不过度承诺")


# ------------------------------------------------------------ D12 临床计算与工具

def build_d12():
    df = pd.read_csv(RAW / "MedCalcBench" / "test_data_11_18_final.csv")
    out = []
    for _, r in spread(list(df.iterrows()), 25):
        out.append(item(
            "D12", "临床计算与工具", 0, task_type="calculation",
            question=clean(r["Question"]), src_key="MedCalcBench",
            context=clean(r["Patient Note"]),
            reference_answer=clean(r["Ground Truth Answer"]),
            key_points=[clean(r.get("Ground Truth Explanation"))[:400]],
            scoring={"method": "数值/选项匹配（允许容差）", "metrics": ["accuracy"]},
            capability="从病历里取数并按医学公式计算（需调工具而非心算）",
            difficulty="medium", extra_source="MedCalc-Bench test",
            notes=f"计算器：{clean(r.get('Calculator Name'))}／{clean(r.get('Category'))}",
        ))
    return out


# ------------------------------------------------------------ D13 证据检索与溯源

def build_d13():
    pq = pd.read_parquet(RAW / "PubMedQA" / "pqa_labeled" / "train-00000-of-00001.parquet")
    out = []
    for _, r in spread(list(pq.iterrows()), 20):
        ctx = r["context"]
        if isinstance(ctx, dict):
            contexts = ctx.get("contexts")
            contexts = list(contexts) if contexts is not None else []
            ctx_text = "\n".join(f"[摘要{i + 1}] {clean(c)}" for i, c in enumerate(contexts))
        else:
            ctx_text = clean(ctx)
        # 原文里夹带了少量 HTML 片段（如 </td>），清掉
        ctx_text = re.sub(r"</?[a-zA-Z][^>]{0,20}>", "", ctx_text)
        out.append(item(
            "D13", "证据检索与溯源", 0, task_type="evidence_qa",
            question=clean(r["question"]), src_key="PubMedQA",
            context=ctx_text, answer=clean(r["final_decision"]),
            reference_answer=clean(r["long_answer"]),
            scoring={"method": "结论匹配 + 引用是否指向支撑该结论的摘要",
                     "metrics": ["accuracy", "citation_traceability", "hallucination_rate"]},
            capability="结论必须落到给定证据上，并说清出自哪一条",
            difficulty="medium", extra_source="PubMedQA pqa_labeled",
            notes="标签集合：yes / no / maybe",
        ))
    q = pd.read_parquet(RAW / "CmedqaRetrieval" / "data" / "queries-00000-of-00001-daeedab899d3c839.parquet")
    for _, r in spread(list(q.iterrows()), 5):
        out.append(item(
            "D13", "证据检索与溯源", 0, task_type="retrieval",
            question=clean(r["text"]), src_key="CmedqaRetrieval",
            scoring={"method": "与 _raw 中语料比对，算 Recall@5 / nDCG@10",
                     "metrics": ["recall@5", "ndcg@10"],
                     "corpus": "_raw/CmedqaRetrieval/data/corpus-00000-of-00001-a3949861f65a3226.parquet"},
            capability="中文医学文献检索召回", difficulty="medium",
            extra_source="CmedqaRetrieval queries", notes=f"query id：{r['id']}",
        ))
    return out


# ------------------------------------------------------------------ 中文化

def _path_key(path) -> str:
    return ".".join(str(p) for p in path)


def _needs_translation(text: str) -> bool:
    return len(re.findall(r"[A-Za-z]", text)) >= 8


def localize(items: list, keep_en: int = 0, label: str = "") -> list:
    """把英文题翻成中文，最后 keep_en 道保留英文原题。

    译前原文写进 original_en，翻译缓存落在 _build/translation_cache.json，
    重跑不会重复调用接口。
    """
    cut = len(items) - keep_en if keep_en else len(items)
    todo = items[:cut]
    if not todo:
        return items

    jobs = []
    for it in todo:
        for path, text in iter_strings(it):
            if text and _needs_translation(text):
                jobs.append((it, path, text))
    if jobs:
        tr = Translator()
        translated = tr.to_zh_batch([t for _, _, t in jobs])
        tr.save()
        for (it, path, src), zh in zip(jobs, translated):
            if zh and zh != src:
                set_path(it, path, zh)
            it.setdefault("original_en", {})[_path_key(path)] = src
        print(f"    {label} 翻译 {len(jobs)} 段（累计接口调用 {tr.calls} 次）")

    for it in todo:
        it["language"] = "zh"
        it["translation"] = {
            "translated": True,
            "model": "deepseek-chat",
            "direction": "en → zh",
            "note": "机器翻译，正式使用前建议由医学背景人员校对；原文见 original_en 字段。",
        }
    return items


BUILDERS = [
    ("01_医学知识基础.jsonl", build_d01, 5),
    ("02_病例诊断推理.jsonl", build_d02, 2),
    ("03_鉴别诊断.jsonl", build_d03, 5),
    ("04_检查决策.jsonl", build_d04, 0),
    ("05_诊断分析_开放题.jsonl", build_d05, 0),
    ("06_多轮问诊诊断.jsonl", build_d06, 0),
    ("07_主动问诊_智能体.jsonl", build_d07, 3),
    ("08_疑难病例_全流程.jsonl", build_d08, 5),
    ("09_安全与急症边界.jsonl", build_d09, 5),
    ("10_信息不足与不确定性.jsonl", build_d10, 4),
    ("11_医患沟通与解释.jsonl", build_d11, 4),
    ("12_临床计算与工具.jsonl", build_d12, 5),
    ("13_证据检索与溯源.jsonl", build_d13, 5),
]


def main():
    random.seed(SEED)
    manifest, all_items = [], []
    for filename, fn, keep_en in BUILDERS:
        print(f"  构建 {filename}")
        items = localize(fn(), keep_en=keep_en, label=filename[:2])
        write_items(items, filename)
        all_items += items
        manifest.append({
            "file": filename,
            "dimension_code": items[0]["dimension_code"] if items else None,
            "dimension": items[0]["dimension"] if items else None,
            "n_items": len(items),
            "n_translated": sum(1 for i in items if i.get("translation")),
            "task_types": sorted({i["task_type"] for i in items}),
            "language": dict(collections.Counter(i["language"] for i in items)),
            "sources": sorted({i["source"]["name"] for i in items}),
        })
    lang = collections.Counter(i["language"] for i in all_items)
    (ROOT / "manifest.json").write_text(
        json.dumps({"collected_at": all_items[0]["provenance"]["collected_at"],
                    "seed": SEED,
                    "total": len(all_items),
                    "language": dict(lang),
                    "dimensions": manifest},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n合计 {len(all_items)} 题，{len(manifest)} 个维度；"
          f"中文 {lang.get('zh', 0)} 题、英文 {lang.get('en', 0)} 题")


if __name__ == "__main__":
    main()
