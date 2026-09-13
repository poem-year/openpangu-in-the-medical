# -*- coding: utf-8 -*-
"""面向医学诊断场景的测评集 v3（最终版）
K1 病例诊断选择题 75 (CMExam) | K2 诊断性检查选择题 15 (CMExam)
K3 病例诊断分析 16 (CMB-Clin) | K4 鉴别诊断 24 (CMB-Clin) | K5 多轮问诊诊断结论 20 (IMCS-V2)
共 150 题，全部只考"诊断"。
"""
import json, io, os, re, csv, shutil
from collections import Counter

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

review = io.StringIO()
items = []
def s(x): return x if isinstance(x, str) else ("" if x is None else str(x))
def flat(x): return re.sub(r"\s+", "", s(x))
def clean(x): return re.sub(r"\s+", " ", s(x)).strip(" 　-—、，,。；;：:")
def add(it, lv, mt):
    it["level_hint"] = lv; it["metrics"] = mt; items.append(it)
def spread(pool, n):
    if len(pool) <= n: return pool
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]

VIG = re.compile(r"(男|女|患儿|患者|男孩|女孩|孕妇|婴儿|某)[，,]?\s*\d+|[0-9]+\s*(岁|个月|天)的?(男|女|患儿|患者)|患者[，,：:]|患儿[，,：:]")
ASK_DX = re.compile(r"最可能的诊断|最可能为|应首先考虑|首先考虑|诊断是|诊断为|最可能诊断|最可能是")
ASK_WU = re.compile(r"首选.*检查|最有助于确诊|为明确诊断|明确诊断|最有价值.*检查|有价值的检查|首选.*方法|应做.*检查|进一步检查|首选的检查|最有帮助")
NEG = re.compile(r"除外|不包括|错误|不正确|不宜|无需|不必要|不需要")
NEG_TX = re.compile(r"首选治疗|治疗方案|治疗措施|用药|药引|服药|给药|药物有|药剂|方剂|中药|中成药|舌苔|脉弦|脉细|脉沉|证型")
OPT_EXAM = re.compile(r"检查|检验|试验|镜|CT|MRI|X线|超声|造影|培养|活检|穿刺|刮宫|心电图|脑电图|血气|测定")
OPT_TCM = re.compile(r"证$|虚|瘀|湿热|风寒|痰|阴虚|阳虚|气虚|血虚")
PREF = re.compile(r"首选|最有助于确诊|最有价值|为明确诊断|最有帮助")

# ---------- K1 / K2 ----------
cm = []
for fn in ["CMExam_valid.json", "CMExam_test.json"]:
    for line in open(os.path.join(RAW, fn), encoding="utf-8"):
        if line.strip(): cm.append(json.loads(line))
pool_dx, pool_wu, seen = [], [], set()
for it in cm:
    q = s(it.get("Question")); opts = {o["key"]: o["value"] for o in it.get("Options", [])}
    if len(opts) < 4 or not VIG.search(q) or NEG_TX.search(q + " ".join(opts.values())):
        continue
    stem = flat(q)[:40]
    if stem in seen: continue
    otext = " ".join(opts.values())
    tail = q[-45:]                      # 只看问句结尾，避免题干中间出现“诊断为”
    if re.search(r"治疗|用药|药物|抗菌|镇痛|处理|护理|手术|给药|预后|并发症|预防|接种|饮食|注射|给予|疫苗|麻醉|输血", tail):
        continue
    if ASK_WU.search(tail) and not NEG.search(tail) and OPT_EXAM.search(otext) and not OPT_TCM.search(otext):
        seen.add(stem); pool_wu.append(it)
    elif ASK_DX.search(tail):
        seen.add(stem); pool_dx.append(it)
pool_wu.sort(key=lambda x: 0 if PREF.search(s(x.get("Question"))) else 1)
print(f"[K1/K2] CMExam {len(cm)} 题 -> 诊断池 {len(pool_dx)}，检查池 {len(pool_wu)}", file=review)

def mcq(i, it, prefix, cat, task, scoring):
    add({"id": f"{prefix}-{i:03d}", "category": cat, "subcategory": "国家执业医师资格考试",
         "task": task, "question": s(it.get("Question")),
         "options": {o["key"]: o["value"] for o in it.get("Options", [])},
         "answer": s(it.get("Answer")), "explanation": s(it.get("Explanation")),
         "source": SRC["CMExam"], "scoring": scoring},
        ["L0", "L1", "L2", "L3"], ["准确率"])

for i, it in enumerate(spread(pool_dx, 75), 1):
    mcq(i, it, "DX1", "病例诊断选择题", "diagnosis_mcq", "选项精确匹配；解析用于核对诊断依据")
for i, it in enumerate(spread(pool_wu, 15), 1):
    mcq(i, it, "DX2", "诊断性检查选择题", "workup_mcq", "选项精确匹配；解析用于核对检查指征")

# ---------- K3 / K4 ----------
clin = json.load(open(os.path.join(RAW, "CMB-Clin-qa.json"), encoding="utf-8"))
OPTMARK = re.compile(r"(?:^|\n)\s*[A-E][.、．]\s*\S")
BAD_DX = re.compile(r"^[（(①-⑩]|^青年|^中年|^老年|^儿童|^病人|^患者|^患儿|^该|^本例|^本病|^临床|依据|症状|体征|病史|检查|检验|治疗|建议|原则|解析|应|需|注意|包括|主诉|观察|随诊|诊断[一二三四五六七八九十]")
BAD_DDX = re.compile(r"可见|可鉴|可以|应|需|会|出现|表现|提示|不符|不能|包括|目前|待|是否|其中|多以|易于|罕见|多见|常|均|多伴|"
                     r"本例|该例|此例|本病例|病人|患者|若|则|但|等$|鉴别$|诊断$|诊断依据|实验室检查|体格检查|辅助检查|病史|症状|体征|治疗|建议|原则|方法|标准|分型|要点")

def extract_dx(ans):
    ans = s(ans).strip(); cands = []
    for m in re.finditer(r"诊断[：:]\s*([^\n]{2,90})", ans):
        cands.append(re.split(r"[。；;]", m.group(1))[0])
    for pat in [r"(?:初步|入院|最后|最终)?诊断为\s*([^\n，,。；;]{2,45})",
                r"初步诊断[：:]\s*([^\n]{2,60})",
                r"首先考虑的诊断(?:应该)?是\s*([^\n，,。；;]{2,45})"]:
        m = re.search(pat, ans)
        if m: cands.append(m.group(1))
    cands.append(re.split(r"[。；;\n]", ans)[0])          # 首句兜底
    out, seend = [], set()
    for c in cands:
        c = clean(c)
        c = re.sub(r"^(?:诊断为|诊断|应诊断为)\s*[：:]?\s*", "", c)
        c = re.sub(r"^[①-⑩、\s]+", "", c)
        if c.count("（") > c.count("）"): c = c[:c.rfind("（")].strip()
        k = flat(c)
        if 3 <= len(c) <= 45 and not BAD_DX.search(c) and not OPTMARK.search(c) and k not in seend:
            seend.add(k); out.append(c)
    if out: return out[:2]
    nums = [clean(n) for n in re.findall(r"[①-⑩]\s*([^；;。\n]{2,40})", ans)]
    nums = [n for n in nums if 3 <= len(n) <= 40 and not BAD_DX.search(n)]
    return nums[:3] if len(nums) >= 2 else []

def extract_ddx(ans):
    ans = s(ans).strip()
    m = re.search(r"鉴别诊断\s*[：:]?", ans)
    seg = ans[m.end():] if m else ans
    if len(clean(seg)) < 6: seg = ans
    names = []
    for part in re.split(r"[①②③④⑤⑥⑦⑧⑨⑩]|(?:\d+)\s*[、.．)）]|[（(]\d+[)）]|\n", seg):
        head = clean(re.split(r"[：:，,。；;（(]", part.strip())[0])
        if 3 <= len(head) <= 26 and not BAD_DDX.search(head) and "鉴别" not in head:
            names.append(head)
    if len(names) < 2:
        m2 = re.search(r"鉴别(?:诊断)?[：:]?\s*([^\n。]{4,240})", ans)
        if m2:
            cand = [clean(c) for c in re.split(r"[、，,；;]", m2.group(1))]
            cand = [c for c in cand if 3 <= len(c) <= 26 and not BAD_DDX.search(c)]
            if len(cand) >= 2: names = cand
    out, seenn = [], set()
    for n in names:
        k = flat(n)
        if k not in seenn: seenn.add(k); out.append(n)
    return out[:8]

dx_pool, ddx_pool = [], []
for case in clin:
    for qa in case.get("QA_pairs", []):
        q = s(qa.get("question"))
        if "鉴别" in q:
            kp = extract_ddx(qa.get("answer"))
            if kp: ddx_pool.append((case, qa, kp))
        elif not OPTMARK.search(q) and not re.search(r"治疗|处理|原则|方案|手术", q) and re.search(r"诊断|最可能|确诊|判断", q):
            kp = extract_dx(qa.get("answer"))
            if kp: dx_pool.append((case, qa, kp))
print(f"[K3/K4] CMB-Clin 可信诊断题 {len(dx_pool)}（覆盖 {len(set(c.get('id') for c,_,_ in dx_pool))} 例），"
      f"鉴别题 {len(ddx_pool)}（覆盖 {len(set(c.get('id') for c,_,_ in ddx_pool))} 例）", file=review)

def pick(pool, n):
    ordered = sorted(pool, key=lambda t: (0 if "诊断依据" in s(t[1].get("question")) else 1, len(s(t[1].get("question")))))
    out, seen = [], set()
    for case, qa, kp in ordered:
        cid = s(case.get("id"))
        if cid in seen: continue
        seen.add(cid); out.append((case, qa, kp))
        if len(out) >= n: break
    return out

for i, (case, qa, kp) in enumerate(pick(dx_pool, 16), 1):
    add({"id": f"DX3-{i:03d}", "category": "病例诊断分析", "subcategory": s(case.get("title")),
         "task": "case_diagnosis", "case_id": s(case.get("id")), "case_title": s(case.get("title")),
         "context": s(case.get("description")), "question": s(qa.get("question")),
         "reference_answer": s(qa.get("answer")), "key_points": ["诊断：" + k for k in kp],
         "key_points_note": "自动抽取，正式评分前需医学人员校核",
         "source": SRC["CMB-Clin"], "scoring": "主诊断是否命中 + 诊断依据覆盖度 + 2 名评分人 1-5 分"},
        ["L2", "L3", "L4"], ["准确率", "引用可追溯率", "幻觉率"])

for i, (case, qa, kp) in enumerate(pick(ddx_pool, 26), 1):
    add({"id": f"DX4-{i:03d}", "category": "鉴别诊断", "subcategory": s(case.get("title")),
         "task": "differential_diagnosis", "case_id": s(case.get("id")), "case_title": s(case.get("title")),
         "context": s(case.get("description")), "question": s(qa.get("question")),
         "reference_answer": s(qa.get("answer")), "key_points": ["鉴别：" + k for k in kp],
         "key_points_note": "自动抽取，正式评分前需医学人员校核",
         "source": SRC["CMB-Clin"], "scoring": "应鉴别疾病召回率 + 区分要点合理性 + 2 名评分人 1-5 分"},
        ["L2", "L3", "L4"], ["准确率", "引用可追溯率", "幻觉率"])

# ---------- K5 ----------
lines = [json.loads(l) for l in open(os.path.join(RAW, "PromptCBLUE_dev.json"), encoding="utf-8") if l.strip()]
mrg = [x for x in lines if x.get("task_dataset") == "IMCS-V2-MRG"]
def parse_dialogue(text):
    body = text.split("问诊对话历史：", 1)
    body = body[1] if len(body) > 1 else text
    return [{"role": m.group(1), "text": m.group(2).strip()} for m in re.finditer(r"(医生|患者)：(.+)", body)]
for i, it in enumerate(spread(mrg, 20), 1):
    rep = s(it.get("target"))
    kps = [l.strip() for l in rep.split("\n") if re.match(r"^(诊断|建议)", l.strip())]
    add({"id": f"DX5-{i:03d}", "category": "多轮问诊诊断结论", "subcategory": "医患多轮对话",
         "task": "diagnosis_from_dialogue", "dialogue": parse_dialogue(s(it.get("input"))),
         "question": "阅读上述完整问诊对话，给出你的诊断结论：先列出最可能的诊断，再列出需要鉴别的疾病，并说明判断依据；信息不足时要明确指出还需补充哪些信息。",
         "reference_answer": rep, "key_points": kps or [rep.strip()[:90]],
         "key_points_note": "自动抽取，正式评分前需医学人员校核",
         "source": SRC["IMCS-V2"], "scoring": "诊断与参考一致性 + 依据与不确定性说明 + 2 名评分人 1-5 分"},
        ["L1", "L4", "L5"], ["准确率", "安全合规率"])

# ---------- 输出 ----------
old = os.path.join(ROOT, "_superseded_v1"); os.makedirs(old, exist_ok=True)
for f in ["medical_diagnosis_eval.jsonl", "medical_diagnosis_eval_questions.jsonl", "answer_key.csv"]:
    p = os.path.join(ROOT, f)
    if os.path.exists(p) and not os.path.exists(os.path.join(old, f)):
        shutil.move(p, os.path.join(old, f))
with open(os.path.join(ROOT, "medical_diagnosis_eval.jsonl"), "w", encoding="utf-8") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
STRIP = {"answer", "reference_answer", "explanation", "key_points", "key_points_note", "scoring"}
with open(os.path.join(ROOT, "medical_diagnosis_eval_questions.jsonl"), "w", encoding="utf-8") as f:
    for it in items: f.write(json.dumps({k: v for k, v in it.items() if k not in STRIP}, ensure_ascii=False) + "\n")
with open(os.path.join(ROOT, "answer_key.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["题号", "类别", "子类/病例", "参考答案", "评分方式", "出处数据集"])
    for it in items:
        ans = it.get("answer") or "；".join(it.get("key_points") or []) or it.get("reference_answer", "")[:80]
        w.writerow([it["id"], it["category"], it.get("case_title") or it["subcategory"], ans, it["scoring"], it["source"]["dataset"]])

print(file=review)
print("总题量:", len(items), file=review)
print(Counter(i["category"] for i in items), file=review)
print("出处:", Counter(i["source"]["dataset"] for i in items), file=review)
open(os.path.join(RAW, "_review3.txt"), "w", encoding="utf-8").write(review.getvalue())
print("items:", len(items))