# -*- coding: utf-8 -*-
"""v2 修订：① 更稳的诊断/鉴别要点抽取 ② 检查题改为"首选/最有价值"型，去掉负向题。"""
import json, os, re, csv
from collections import Counter

ROOT = r"E:\面向医疗诊断场景的openpangu能力增强方案\评测集"
RAW = os.path.join(ROOT, "_raw")
def s(x): return x if isinstance(x, str) else ("" if x is None else str(x))
def clean(x):
    x = re.sub(r"\s+", " ", s(x)).strip(" 　-—:：、，,。；;")
    return x[:70]

STOP = re.compile(r"诊断依据|可排除|表现为|依据|辅助检查|查体|病史|症状|体征|建议|治疗")

def split_names(seg):
    seg = re.sub(r"[①-⑩]", "|", seg)
    seg = re.sub(r"\d+\s*[、.．)）]", "|", seg)
    parts = re.split(r"[|、，,；;。]", seg)
    out = []
    for p in parts:
        p = clean(p)
        if 2 <= len(p) <= 30 and not STOP.search(p) and not re.search(r"[（(]|：", p):
            out.append(p)
    return out

def kp_dx(ans):
    ans = s(ans).strip()
    m = re.search(r"诊断[：:为]\s*([^\n]{2,90})", ans)
    if m:
        d = clean(re.split(r"[。；;]", m.group(1))[0])
        if d.count("（") > d.count("）"): d = d[:d.rfind("（")].strip()
        if d: return ["诊断：" + d]
    m = re.search(r"初步诊断为\s*([^\n，,。；;]{2,40})", ans)
    if m:
        return ["诊断：" + clean(m.group(1))]
    m = re.search(r"诊断为\s*([^\n，,。；;]{2,40})", ans)
    if m:
        return ["诊断：" + clean(m.group(1))]
    # 分条列出的多个诊断：（1）急性梗阻性化脓性胆管炎 / 1、xxx
    nums = re.findall(r"[（(]\d+[)）]\s*([^\n。；;]{2,40})", ans)
    nums = [clean(n) for n in nums if 2 <= len(clean(n)) <= 40 and not STOP.search(n)]
    if len(nums) >= 2:
        return ["诊断：" + n for n in nums[:5]]
    head = clean(re.split(r"[。；;\n]", ans)[0])
    return ["诊断：" + head] if len(head) >= 4 else ([clean(ans[:70])] if ans else [])

def kp_ddx(ans):
    ans = s(ans).strip()
    seg = ans
    m = re.search(r"鉴别诊断[：:]?\s*([\s\S]{0,300})", ans)
    if m: seg = m.group(1)
    else:
        m = re.search(r"鉴别\s*([\s\S]{0,300})", ans)
        if m: seg = m.group(1)
    names = split_names(seg)
    if not names:
        names = split_names(ans)
    return ["鉴别：" + n for n in names[:8]]

# ---------- 修要点 ----------
p = os.path.join(ROOT, "medical_diagnosis_eval.jsonl")
items = [json.loads(l) for l in open(p, encoding="utf-8")]
fixed = 0
for it in items:
    if it["task"] == "case_diagnosis":
        new = kp_dx(it["reference_answer"])
    elif it["task"] == "differential_diagnosis":
        new = kp_ddx(it["reference_answer"])
    else:
        continue
    if new != it["key_points"]:
        fixed += 1
    it["key_points"] = new

# ---------- 重做 K2 检查题 ----------
VIG = re.compile(r"(男|女|患儿|患者|男孩|女孩|孕妇|婴儿|某)[，,]?\s*\d+|[0-9]+\s*(岁|个月|天)的?(男|女|患儿|患者)|患者[，,：:]|患儿[，,：:]")
ASK_WORKUP = re.compile(r"首选.*检查|最有助于确诊|为明确诊断|明确诊断|最有价值.*检查|有价值的检查|首选.*方法|应做.*检查|进一步检查|首选的检查|最有帮助")
NEG = re.compile(r"除外|不包括|错误|不正确|不宜|无需")
NEG_TX = re.compile(r"首选治疗|治疗方案|治疗措施|用药|药引|服药|给药|药物有|药剂|方剂|中药|中成药|舌苔|脉弦|脉细|脉沉|证型")
OPTS_EXAM = re.compile(r"检查|检验|试验|镜|CT|MRI|X线|超声|造影|培养|活检|穿刺|刮宫|心电图|脑电图|血气|测定")
OPTS_TCM = re.compile(r"证$|虚|瘀|湿热|风寒|痰|阴虚|阳虚|气虚|血虚")

cm = []
for fn in ["CMExam_valid.json", "CMExam_test.json"]:
    for line in open(os.path.join(RAW, fn), encoding="utf-8"):
        if line.strip(): cm.append(json.loads(line))

used = {re.sub(r"\s", "", it["question"])[:40] for it in items if it.get("options")}
PREF = re.compile(r"首选|最有助于确诊|最有价值|为明确诊断|最有帮助")
pool, seen = [], set()
for it in cm:
    q = s(it.get("Question")); opts = {o["key"]: o["value"] for o in it.get("Options", [])}
    if len(opts) < 4 or not VIG.search(q) or NEG.search(q) or NEG_TX.search(q) or NEG_TX.search(" ".join(opts.values())):
        continue
    if re.sub(r"\s", "", q)[:40] in used or not ASK_WORKUP.search(q):
        continue
    otext = " ".join(opts.values())
    if not OPTS_EXAM.search(otext) or OPTS_TCM.search(otext):
        continue
    stem = re.sub(r"\s", "", q)[:40]
    if stem in seen: continue
    seen.add(stem); pool.append(it)
pool.sort(key=lambda x: 0 if PREF.search(s(x.get("Question"))) else 1)
print("检查题池(去负向/去治疗):", len(pool))
step = len(pool) / 15
sel = [pool[int(i * step)] for i in range(15)]

new_k2 = []
for i, it in enumerate(sel, 1):
    base = [x for x in items if x["id"] == f"DX2-{i:03d}"][0]
    new_k2.append({**base,
        "question": s(it.get("Question")),
        "options": {o["key"]: o["value"] for o in it.get("Options", [])},
        "answer": s(it.get("Answer")), "explanation": s(it.get("Explanation"))})
items = [it for it in items if not it["id"].startswith("DX2-")] + new_k2
items.sort(key=lambda x: x["id"])

# ---------- 重写输出 ----------
with open(p, "w", encoding="utf-8") as f:
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

print("修正要点题数:", fixed)
print("要点仍为空的开放题:", [i["id"] for i in items if not i.get("options") and not i.get("key_points")])
print("K2 样例题干:")
for i in new_k2[:5]: print("   ", i["question"][:70], "=>", i["answer"])
print("总题量:", len(items), Counter(i["category"] for i in items))