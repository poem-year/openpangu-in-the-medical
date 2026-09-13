# -*- coding: utf-8 -*-
"""从公开权威中文医疗测评集抽样，组装面向医疗诊断场景的大模型测评集（最终版）。
A 医学知识选择题(CMB-Exam val) 50 | B 医学知识选择题(CMExam) 30
C 临床病例分析(CMB-Clin) 30 | D 多轮问诊诊疗报告(IMCS-V2) 20 | E 开放式医学问答(cMedQA-V2.0) 20
"""
import json, io, os, re, csv, urllib.request
from collections import Counter, defaultdict

ROOT = r"E:\面向医疗诊断场景的openpangu能力增强方案\评测集"
RAW = os.path.join(ROOT, "_raw")

SRC = {
    "CMB-Exam": {"dataset": "CMB-Exam（验证集）", "full_name": "CMB · Chinese Medical Benchmark 医学考试题",
        "org": "FreedomIntelligence（上海交通大学 / 华东师范大学）",
        "url": "https://modelscope.cn/datasets/FreedomIntelligence/CMB",
        "url_alt": "https://huggingface.co/datasets/FreedomIntelligence/CMB",
        "license": "Apache-2.0", "paper": "https://arxiv.org/abs/2308.08833"},
    "CMB-Clin": {"dataset": "CMB-Clin", "full_name": "CMB · 临床病例分析（住院病历级案例）",
        "org": "FreedomIntelligence（上海交通大学 / 华东师范大学）",
        "url": "https://modelscope.cn/datasets/FreedomIntelligence/CMB",
        "url_alt": "https://huggingface.co/datasets/FreedomIntelligence/CMB",
        "license": "Apache-2.0", "paper": "https://arxiv.org/abs/2308.08833"},
    "CMExam": {"dataset": "CMExam", "full_name": "CMExam 国家执业医师资格考试题（含解析）", "org": "中山大学等",
        "url": "https://huggingface.co/datasets/fzkuji/CMExam", "url_alt": "",
        "license": "Apache-2.0", "paper": "https://arxiv.org/abs/2306.03030"},
    "IMCS-V2": {"dataset": "IMCS-V2-MRG", "full_name": "IMCS-V2 医患多轮问诊 · 诊疗报告生成（PromptCBLUE / CHIP-2023）",
        "org": "CBLUE 中文医疗信息处理评测 / CHIP 会议",
        "url": "https://huggingface.co/datasets/tchenglv/PromptCBLUE", "url_alt": "",
        "license": "MIT（PromptCBLUE 封装版本）", "paper": "https://arxiv.org/abs/2304.14389"},
    "cMedQA-V2.0": {"dataset": "cMedQA-V2.0", "full_name": "cMedQA-V2.0 中文医学问答（患者提问 + 医生回答）",
        "org": "cMedQA2 原始数据集（Zhang et al., IEEE Access 2018）",
        "url": "https://huggingface.co/datasets/wangrongsheng/cMedQA-V2.0", "url_alt": "",
        "license": "未标注（学术研究用途）", "paper": "https://ieeexplore.ieee.org/document/8279564"},
}

LEVEL_HINT = {
    "medical_knowledge": (["L0", "L1", "L2"], ["准确率"]),
    "case_analysis": (["L2", "L3", "L4"], ["准确率", "引用可追溯率"]),
    "multi_turn": (["L1", "L4"], ["准确率", "人工评分"]),
    "open_qa": (["L0", "L1", "L5"], ["准确率", "安全合规率", "人工评分"]),
}

review = io.StringIO()
items = []
def s(x): return x if isinstance(x, str) else ("" if x is None else str(x))
def norm_q(x): return re.sub(r"[\s　,，。.、；;：:（）()【】\[\]“”\"'’‘]", "", s(x)).lower()
def is_chinese(x): return len(re.findall(r"[\u4e00-\u9fff]", s(x))) >= 8

def add(it, kind):
    lv, mt = LEVEL_HINT[kind]
    it["level_hint"] = lv
    it["metrics"] = mt
    items.append(it)

# ================= A. CMB-Exam val =================
val = json.load(open(os.path.join(RAW, "CMB", "CMB-Exam", "CMB-val", "CMB-val-merge.json"), encoding="utf-8"))
CLIN_WEIGHT = {"临床医学": 3, "执业医师": 3, "执业助理": 3, "西医综合": 3, "规培结业": 3,
               "中级职称": 2, "高级职称": 2, "医学考研": 2, "护理学": 2, "医技": 2,
               "主管护师": 2, "护士执业资格": 2, "护师执业资格": 2,
               "基础医学": 1, "预防医学与公共卫生学": 1}
DIAG_KW = re.compile(r"最可能|诊断|首先考虑|应考虑|应首先|鉴别|首选|典型|表现|见于|提示|确诊")
TREAT_KW = re.compile(r"治疗|处理|用药|手术|急救")

pool, seen_q = [], set()
for it in val:
    q = s(it.get("question"))
    if it.get("exam_class") in {"考研政治"}:
        continue
    if re.search(r"中药|中医", s(it.get("exam_class")) + s(it.get("exam_subject"))):
        continue
    if not is_chinese(q) or norm_q(q) in seen_q:
        continue
    seen_q.add(norm_q(q))
    w = CLIN_WEIGHT.get(it.get("exam_class"), 1)
    pool.append((w * 4 + (3 if DIAG_KW.search(q) else 0) + (1 if TREAT_KW.search(q) else 0), it))
pool.sort(key=lambda x: -x[0])

sel, per_class, per_stem = [], Counter(), Counter()
for score, it in pool:
    c, stem = it.get("exam_class"), norm_q(it["question"])[:40]
    if per_class[c] >= 6 or per_stem[stem] >= 2:
        continue
    sel.append(it); per_class[c] += 1; per_stem[stem] += 1
    if len(sel) >= 50:
        break
print(f"[A] CMB-Exam val: 原始 {len(val)} -> 过滤去重 {len(pool)} -> 抽 {len(sel)}", file=review)
print(f"    考试类别分布: {dict(per_class)}", file=review)

for i, it in enumerate(sel, 1):
    add({"id": f"MD-A-{i:03d}", "category": "医学知识选择题",
         "subcategory": f"{s(it.get('exam_class'))}/{s(it.get('exam_subject'))}",
         "task": "multi_choice" if len(s(it.get("answer"))) > 1 else "single_choice",
         "question": s(it.get("question")), "options": it.get("option", {}) or {},
         "answer": s(it.get("answer")), "explanation": s(it.get("explanation")),
         "source": SRC["CMB-Exam"], "scoring": "选择题按选项精确匹配，多选须完全一致"}, "medical_knowledge")

# ================= B. CMExam =================
cm = []
for fn in ["CMExam_valid.json", "CMExam_test.json"]:
    for line in open(os.path.join(RAW, fn), encoding="utf-8"):
        if line.strip():
            cm.append(json.loads(line))
DIAG_KW2 = re.compile(r"最可能|诊断|首先考虑|应首先|鉴别|首选|典型|应考虑")
cand = [it for it in cm
        if norm_q(it.get("Question")) not in seen_q and is_chinese(it.get("Question"))
        and DIAG_KW2.search(s(it.get("Question")))]
step = max(1, len(cand) // 30)
sel_b = [cand[i * step] for i in range(30)] if len(cand) >= 30 else cand
print(f"[B] CMExam: 载入 {len(cm)} -> 诊断类且不重复 {len(cand)} -> 抽 {len(sel_b)}", file=review)
for i, it in enumerate(sel_b, 1):
    add({"id": f"MD-B-{i:03d}", "category": "医学知识选择题", "subcategory": "国家执业医师资格考试",
         "task": "single_choice", "question": s(it.get("Question")),
         "options": {o["key"]: o["value"] for o in it.get("Options", [])},
         "answer": s(it.get("Answer")), "explanation": s(it.get("Explanation")),
         "source": SRC["CMExam"], "scoring": "选择题按选项精确匹配"}, "medical_knowledge")

# ================= C. CMB-Clin =================
clin = json.load(open(os.path.join(RAW, "CMB-Clin-qa.json"), encoding="utf-8"))
CASE_KW = re.compile(r"诊断|鉴别|检查|治疗|处理|原则|下一步|方案")
cand_c = []
for case in clin:
    picked = [qa for qa in case.get("QA_pairs", []) if CASE_KW.search(s(qa.get("question")))]
    for qa in picked[:2]:
        cand_c.append((case, qa))
step = max(1, len(cand_c) // 30)
sel_c = [cand_c[i * step] for i in range(30)] if len(cand_c) >= 30 else cand_c
seen_case, sel_c2 = Counter(), []
for case, qa in sel_c:
    cid = s(case.get("id"))
    if seen_case[cid] >= 2:
        continue
    seen_case[cid] += 1
    sel_c2.append((case, qa))
sel_c = sel_c2[:30]
print(f"[C] CMB-Clin: {len(clin)} 例 -> 可选题 {len(cand_c)} -> 抽 {len(sel_c)}（覆盖 {len(seen_case)} 例病例）", file=review)

def key_points_from(question, answer):
    ans = s(answer).strip()
    kps = []
    m = re.search(r"诊断[：:]\s*([^\n]{2,90})", ans)
    if m:
        diag = re.split(r"[。；;]", m.group(1))[0].strip(" （）()")
        if diag:
            kps.append("诊断：" + diag)
    if "鉴别" in s(question):
        tail = re.split(r"鉴别诊断[：:]", ans)
        seg = tail[1] if len(tail) > 1 else ans
        names = re.findall(r"[①②③④⑤⑥⑦⑧⑨⑩]\s*([^：:\n，,。；;（(]{2,25})[：:]", seg)
        if not names:
            names = re.findall(r"[（(]\d+[)）]\s*([^：:\n，,。；;（(]{2,25})[：:]", seg)
        kps += ["鉴别：" + n.strip() for n in names[:6]]
    if not kps:
        head = re.split(r"[。\n]", ans)[0].strip()
        if len(head) >= 8:
            kps.append(head[:90])
    return kps

for i, (case, qa) in enumerate(sel_c, 1):
    add({"id": f"MD-C-{i:03d}", "category": "临床病例分析", "subcategory": s(case.get("title")),
         "task": "case_analysis", "context": s(case.get("description")),
         "question": s(qa.get("question")), "reference_answer": s(qa.get("answer")),
         "key_points": key_points_from(qa.get("question"), qa.get("answer")),
         "key_points_note": "要点由脚本自动抽取，正式评分前需医学人员校核",
         "source": SRC["CMB-Clin"],
         "scoring": "要点覆盖率 + 2 名评分人 1-5 分独立打分"}, "case_analysis")

# ================= D. IMCS-V2 =================
lines = [json.loads(l) for l in open(os.path.join(RAW, "PromptCBLUE_dev.json"), encoding="utf-8") if l.strip()]
mrg = [x for x in lines if x.get("task_dataset") == "IMCS-V2-MRG"]
step = max(1, len(mrg) // 20)
sel_d = [mrg[i * step] for i in range(20)]
print(f"[D] IMCS-V2-MRG: {len(mrg)} 条 -> 抽 {len(sel_d)}", file=review)

def parse_dialogue(text):
    body = text.split("问诊对话历史：", 1)
    body = body[1] if len(body) > 1 else text
    return [{"role": m.group(1), "text": m.group(2).strip()}
            for m in re.finditer(r"(医生|患者)：(.+)", body)]

for i, it in enumerate(sel_d, 1):
    report = s(it.get("target"))
    kps = [l.strip() for l in report.split("\n")
           if re.match(r"^(诊断|建议|主诉|现病史|辅助检查|既往史)", l.strip())]
    add({"id": f"MD-D-{i:03d}", "category": "多轮问诊诊疗报告", "subcategory": "医患多轮对话",
         "task": "report_generation", "dialogue": parse_dialogue(s(it.get("input"))),
         "question": "根据上述医患问诊对话，生成一份综合诊疗报告，包含主诉、现病史、辅助检查、既往史、诊断、建议六个部分。",
         "reference_answer": report, "key_points": kps or [report.strip()[:90]],
         "key_points_note": "要点由脚本自动抽取，正式评分前需医学人员校核",
         "source": SRC["IMCS-V2"],
         "scoring": "六部分齐全度 + 诊断与参考一致性 + 2 名评分人 1-5 分独立打分"}, "multi_turn")

# ================= E. cMedQA-V2.0 =================
def hf_rows(offset, length):
    url = ("https://datasets-server.huggingface.co/rows?dataset=wangrongsheng%2FcMedQA-V2.0"
           f"&config=default&split=train&offset={offset}&length={length}")
    r = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(r, timeout=60)).get("rows", [])

picked_e, seen_e, off = [], set(), 8000
while len(picked_e) < 20 and off < 160000:
    for it in hf_rows(off, 20):
        row = it.get("row", {})
        q, a = s(row.get("instruction")), s(row.get("output"))
        key = norm_q(q)
        if key in seen_e or not is_chinese(q) or len(a) < 120:
            continue
        if not re.search(r"诊断|可能|考虑|建议|检查|治疗", a):
            continue
        seen_e.add(key)
        picked_e.append(row)
        if len(picked_e) >= 20:
            break
    off += 4000
print(f"[E] cMedQA-V2.0: 去重后抽 {len(picked_e)} 条", file=review)
for i, row in enumerate(picked_e, 1):
    add({"id": f"MD-E-{i:03d}", "category": "开放式医学问答", "subcategory": "患者咨询",
         "task": "open_qa", "question": s(row.get("instruction")),
         "reference_answer": s(row.get("output")), "key_points": [],
         "key_points_note": "医生回答为长文本，建议按覆盖度打分，不设固定要点",
         "source": SRC["cMedQA-V2.0"],
         "scoring": "2 名评分人 1-5 分独立打分（信息准确 / 建议合理 / 不越界诊断）"}, "open_qa")

# ================= 输出 =================
with open(os.path.join(ROOT, "medical_diagnosis_eval.jsonl"), "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

STRIP = {"answer", "reference_answer", "explanation", "key_points", "key_points_note", "scoring"}
with open(os.path.join(ROOT, "medical_diagnosis_eval_questions.jsonl"), "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps({k: v for k, v in it.items() if k not in STRIP}, ensure_ascii=False) + "\n")

with open(os.path.join(ROOT, "answer_key.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["题号", "类别", "子类", "参考答案", "评分方式", "出处数据集"])
    for it in items:
        ans = it.get("answer") or (it.get("key_points") or [""])[0] or "见参考答"
        w.writerow([it["id"], it["category"], it["subcategory"], ans, it["scoring"], it["source"]["dataset"]])

print(file=review)
print("== 汇总 ==", file=review)
print("总题量:", len(items), file=review)
print(Counter(i["category"] for i in items), file=review)
print("重复题干检查:", [k for k, v in Counter(i["question"][:40] for i in items).items() if v > 2 and not k.startswith("根据上述")], file=review)
print("缺答案题数:", sum(1 for i in items if not i.get("answer") and not i.get("reference_answer")), file=review)
open(os.path.join(RAW, "_review.txt"), "w", encoding="utf-8").write(review.getvalue())
print("items:", len(items))