# -*- coding: utf-8 -*-
"""面向医学诊断场景的测评集 v2：只保留真正考"诊断"的题。
K1 病例诊断选择题(CMExam) 60 | K2 诊断性检查选择题(CMExam) 15
K3 病例诊断分析(CMB-Clin) 25 | K4 鉴别诊断(CMB-Clin) 20 | K5 多轮问诊→诊断结论(IMCS-V2) 20
"""
import json, io, os, re, csv, shutil
from collections import Counter, defaultdict

ROOT = r"E:\面向医疗诊断场景的openpangu能力增强方案\评测集"
RAW = os.path.join(ROOT, "_raw")

SRC = {
    "CMExam": {"dataset": "CMExam", "full_name": "CMExam 中国国家执业医师资格考试 · 病例型诊断题",
        "org": "中山大学等", "url": "https://huggingface.co/datasets/fzkuji/CMExam", "url_alt": "",
        "license": "Apache-2.0", "paper": "https://arxiv.org/abs/2306.03030"},
    "CMB-Clin": {"dataset": "CMB-Clin", "full_name": "CMB · 临床病例分析（真实住院病历级案例）",
        "org": "FreedomIntelligence（上海交通大学 / 华东师范大学）",
        "url": "https://modelscope.cn/datasets/FreedomIntelligence/CMB",
        "url_alt": "https://huggingface.co/datasets/FreedomIntelligence/CMB",
        "license": "Apache-2.0", "paper": "https://arxiv.org/abs/2308.08833"},
    "IMCS-V2": {"dataset": "IMCS-V2-MRG", "full_name": "IMCS-V2 医患多轮问诊 · 诊疗报告生成（PromptCBLUE / CHIP-2023）",
        "org": "CBLUE 中文医疗信息处理评测 / CHIP 会议",
        "url": "https://huggingface.co/datasets/tchenglv/PromptCBLUE", "url_alt": "",
        "license": "MIT（PromptCBLUE 封装版本）", "paper": "https://arxiv.org/abs/2304.14389"},
}

LV = {
    "dx_mcq":  (["L0", "L1", "L2"], ["准确率"]),
    "workup":  (["L1", "L2", "L3"], ["准确率", "引用可追溯率"]),
    "dx_case": (["L2", "L3", "L4"], ["准确率", "引用可追溯率", "幻觉率"]),
    "ddx":     (["L2", "L3", "L4"], ["准确率", "引用可追溯率", "幻觉率"]),
    "dx_dialogue": (["L1", "L4", "L5"], ["准确率", "安全合规率"]),
}

review = io.StringIO()
items = []
def s(x): return x if isinstance(x, str) else ("" if x is None else str(x))
def norm_q(x): return re.sub(r"[\s　,，。.、；;：:（）()【】\[\]“”\"'’‘]", "", s(x)).lower()
def add(it, kind):
    lv, mt = LV[kind]
    it["level_hint"] = lv; it["metrics"] = mt
    items.append(it)

# ---------- K1/K2：CMExam 病例型诊断题 ----------
VIG = re.compile(r"(男|女|患儿|患者|男孩|女孩|孕妇|婴儿|某)[，,]?\s*\d+|[0-9]+\s*(岁|个月|天)的?(男|女|患儿|患者)|患者[，,：:]|患儿[，,：:]")
ASK_DX = re.compile(r"最可能的诊断|最可能为|应首先考虑|首先考虑|诊断是|诊断为|最可能诊断|该患者最可能的|最可能是")
ASK_WORKUP = re.compile(r"首选.*检查|最有助于确诊|为明确诊断|明确诊断.*检查|有价值的检查|首选.*方法|应做.*检查|进一步检查|首选的检查")
NEG_TX = re.compile(r"首选治疗|治疗方案|治疗措施|用药|药引|服药|给药|药物有|药剂|方剂|中药|中成药|舌苔|脉弦|脉细|脉沉|证型")
OPTS_EXAM = re.compile(r"检查|检验|试验|镜|CT|MRI|X线|超声|造影|培养|活检|穿刺|刮宫|心电图|脑电图|血气|抗.*抗体|测定")
OPTS_TCM = re.compile(r"证$|虚|瘀|湿热|风寒|痰|阴虚|阳虚|气虚|血虚")

cm = []
for fn in ["CMExam_valid.json", "CMExam_test.json"]:
    for line in open(os.path.join(RAW, fn), encoding="utf-8"):
        if line.strip():
            cm.append(json.loads(line))
print(f"[K1/K2] CMExam 载入 {len(cm)} 题", file=review)

pool_dx, pool_wu, seen_stem = [], [], set()
for it in cm:
    q = s(it.get("Question")); opts = {o["key"]: o["value"] for o in it.get("Options", [])}
    if len(opts) < 4 or not VIG.search(q):
        continue
    if NEG_TX.search(q) or NEG_TX.search(" ".join(opts.values())):
        continue
    stem = norm_q(q)[:40]
    if stem in seen_stem:
        continue
    otext = " ".join(opts.values())
    if OPTS_EXAM.search(otext) and not OPTS_TCM.search(otext):
        if ASK_WORKUP.search(q):
            seen_stem.add(stem); pool_wu.append(it)
    elif ASK_DX.search(q):
        seen_stem.add(stem); pool_dx.append(it)

print(f"[K1/K2] 病例型诊断题池 {len(pool_dx)}，检查决策题池 {len(pool_wu)}", file=review)

def spread(pool, n):
    if len(pool) <= n: return pool
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]

for i, it in enumerate(spread(pool_dx, 60), 1):
    add({"id": f"DX1-{i:03d}", "category": "病例诊断选择题", "subcategory": "国家执业医师资格考试",
         "task": "diagnosis_mcq", "question": s(it.get("Question")),
         "options": {o["key"]: o["value"] for o in it.get("Options", [])},
         "answer": s(it.get("Answer")), "explanation": s(it.get("Explanation")),
         "source": SRC["CMExam"], "scoring": "选项精确匹配；解析用于核对诊断依据"}, "dx_mcq")

for i, it in enumerate(spread(pool_wu, 15), 1):
    add({"id": f"DX2-{i:03d}", "category": "诊断性检查选择题", "subcategory": "国家执业医师资格考试",
         "task": "workup_mcq", "question": s(it.get("Question")),
         "options": {o["key"]: o["value"] for o in it.get("Options", [])},
         "answer": s(it.get("Answer")), "explanation": s(it.get("Explanation")),
         "source": SRC["CMExam"], "scoring": "选项精确匹配；解析用于核对检查指征"}, "workup")

# ---------- K3/K4：CMB-Clin ----------
clin = json.load(open(os.path.join(RAW, "CMB-Clin-qa.json"), encoding="utf-8"))
dx_pool, ddx_pool = [], []
for case in clin:
    for qa in case.get("QA_pairs", []):
        q = s(qa.get("question"))
        if "鉴别" in q:
            ddx_pool.append((case, qa))
        elif re.search(r"诊断|最可能|确诊|判断", q) and not re.search(r"治疗|处理|原则|方案|手术", q):
            dx_pool.append((case, qa))
print(f"[K3/K4] CMB-Clin 诊断类 {len(dx_pool)}，鉴别类 {len(ddx_pool)}，覆盖病例 "
      f"{len(set(c.get('id') for c,_ in dx_pool))}/{len(set(c.get('id') for c,_ in ddx_pool))}", file=review)

def pick_by_case(pool, n, prefer):
    out, seen = [], set()
    ordered = sorted(pool, key=lambda cq: (0 if prefer in s(cq[1].get("question")) else 1, len(s(cq[1].get("answer")))))
    for case, qa in ordered:
        cid = s(case.get("id"))
        if cid in seen:
            continue
        seen.add(cid); out.append((case, qa))
        if len(out) >= n:
            break
    return out

def key_points_dx(ans):
    ans = s(ans).strip(); kps = []
    m = re.search(r"诊断[：:]\s*([^\n]{2,90})", ans)
    if m:
        d = re.split(r"[。；;]", m.group(1))[0].strip(" 　-—")
        if d.count("（") > d.count("）"): d = d[:d.rfind("（")]
        if d: kps.append("诊断：" + d.strip())
    return kps

def key_points_ddx(ans):
    ans = s(ans).strip()
    tail = re.split(r"鉴别诊断[：:]", ans)
    seg = tail[1] if len(tail) > 1 else ans
    names = re.findall(r"[①②③④⑤⑥⑦⑧⑨⑩]\s*([^：:\n，,。；;（(]{2,25})[：:]", seg)
    if not names:
        names = re.findall(r"[（(]\d+[)）]\s*([^：:\n，,。；;（(]{2,25})[：:]", seg)
    if not names:
        for line in seg.split("\n"):
            head = line.split("：")[0].strip(" 　-—")
            if 2 <= len(head) <= 25 and not re.search(r"[。，,；;？！]", head):
                names.append(head)
    return ["鉴别：" + n.strip() for n in names[:8] if n.strip()]

for i, (case, qa) in enumerate(pick_by_case(dx_pool, 25, "诊断依据"), 1):
    kp = key_points_dx(qa.get("answer"))
    add({"id": f"DX3-{i:03d}", "category": "病例诊断分析", "subcategory": s(case.get("title")),
         "task": "case_diagnosis", "case_id": s(case.get("id")), "case_title": s(case.get("title")),
         "context": s(case.get("description")), "question": s(qa.get("question")),
         "reference_answer": s(qa.get("answer")), "key_points": kp or [],
         "key_points_note": "自动抽取，正式评分前需医学人员校核",
         "source": SRC["CMB-Clin"],
         "scoring": "主诊断是否命中 + 诊断依据覆盖度 + 2 名评分人 1-5 分"}, "dx_case")

for i, (case, qa) in enumerate(pick_by_case(ddx_pool, 20, "鉴别"), 1):
    add({"id": f"DX4-{i:03d}", "category": "鉴别诊断", "subcategory": s(case.get("title")),
         "task": "differential_diagnosis", "case_id": s(case.get("id")), "case_title": s(case.get("title")),
         "context": s(case.get("description")), "question": s(qa.get("question")),
         "reference_answer": s(qa.get("answer")), "key_points": key_points_ddx(qa.get("answer")),
         "key_points_note": "自动抽取，正式评分前需医学人员校核",
         "source": SRC["CMB-Clin"],
         "scoring": "应鉴别疾病召回率 + 区分要点合理性 + 2 名评分人 1-5 分"}, "ddx")

# ---------- K5：IMCS-V2-MRG ----------
lines = [json.loads(l) for l in open(os.path.join(RAW, "PromptCBLUE_dev.json"), encoding="utf-8") if l.strip()]
mrg = [x for x in lines if x.get("task_dataset") == "IMCS-V2-MRG"]
def parse_dialogue(text):
    body = text.split("问诊对话历史：", 1)
    body = body[1] if len(body) > 1 else text
    return [{"role": m.group(1), "text": m.group(2).strip()} for m in re.finditer(r"(医生|患者)：(.+)", body)]

kept = []
for it in mrg:
    rep = s(it.get("target"))
    if re.search(r"诊断[：:]\s*\S", rep):
        kept.append(it)
print(f"[K5] IMCS-V2-MRG 共 {len(mrg)}，其中含明确诊断 {len(kept)}", file=review)
for i, it in enumerate(spread(kept, 20), 1):
    rep = s(it.get("target"))
    kps = [l.strip() for l in rep.split("\n") if re.match(r"^(诊断|建议)", l.strip())]
    add({"id": f"DX5-{i:03d}", "category": "多轮问诊诊断结论", "subcategory": "医患多轮对话",
         "task": "diagnosis_from_dialogue", "dialogue": parse_dialogue(s(it.get("input"))),
         "question": "阅读上述完整问诊对话，给出你的诊断结论（列出最可能的诊断，如有需要鉴别的疾病一并列出），并说明判断依据；信息不足时要说明还需补充什么。",
         "reference_answer": rep, "key_points": kps or [rep.strip()[:90]],
         "key_points_note": "自动抽取，正式评分前需医学人员校核",
         "source": SRC["IMCS-V2"],
         "scoring": "诊断与参考一致性 + 是否说明依据与不确定性 + 2 名评分人 1-5 分"}, "dx_dialogue")

# ---------- 输出 ----------
old = os.path.join(ROOT, "_superseded_v1")
os.makedirs(old, exist_ok=True)
for f in ["medical_diagnosis_eval.jsonl", "medical_diagnosis_eval_questions.jsonl", "answer_key.csv"]:
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        shutil.move(p, os.path.join(old, f))

with open(os.path.join(ROOT, "medical_diagnosis_eval.jsonl"), "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

STRIP = {"answer", "reference_answer", "explanation", "key_points", "key_points_note", "scoring"}
with open(os.path.join(ROOT, "medical_diagnosis_eval_questions.jsonl"), "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps({k: v for k, v in it.items() if k not in STRIP}, ensure_ascii=False) + "\n")

with open(os.path.join(ROOT, "answer_key.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["题号", "类别", "子类/病例", "参考答案", "评分方式", "出处数据集"])
    for it in items:
        ans = it.get("answer") or "；".join(it.get("key_points") or []) or it.get("reference_answer", "")[:80]
        w.writerow([it["id"], it["category"], it.get("case_title") or it["subcategory"], ans, it["scoring"], it["source"]["dataset"]])

print(file=review)
print("== 汇总 ==", file=review)
print("总题量:", len(items), file=review)
print(Counter(i["category"] for i in items), file=review)
print("覆盖病例数: 诊断", len(set(i.get("case_id") for i in items if i["task"]=="case_diagnosis")),
      "鉴别", len(set(i.get("case_id") for i in items if i["task"]=="differential_diagnosis")), file=review)
print("选项数:", Counter(len(i.get("options", {})) for i in items if i.get("options")), file=review)
print("要点为空的开放题:", [i["id"] for i in items if not i.get("options") and not i.get("key_points")], file=review)
open(os.path.join(RAW, "_review2.txt"), "w", encoding="utf-8").write(review.getvalue())
print("items:", len(items))