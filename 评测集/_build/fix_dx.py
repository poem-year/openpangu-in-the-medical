# -*- coding: utf-8 -*-
"""DX3 重做：只保留能抽出可信"诊断名"的题，抽不到就换下一题。"""
import json, os, re, csv

ROOT = r"E:\面向医疗诊断场景的openpangu能力增强方案\评测集"
RAW = os.path.join(ROOT, "_raw")
def s(x): return x if isinstance(x, str) else ("" if x is None else str(x))
def clean(x): return re.sub(r"\s+", " ", s(x)).strip(" 　-—、，,。；;：:")

BAD_DX = re.compile(r"^[（(]|依据|症状|体征|病史|检查|检验|治疗|建议|原则|解析|病人|患者|患儿|本病|应|需|注意|包括|主诉|观察|随诊")
OPTMARK = re.compile(r"(?:^|\n)\s*[A-E][.、．]\s*\S")

def extract_dx(ans):
    ans = s(ans).strip()
    cands = []
    for m in re.finditer(r"诊断[：:]\s*([^\n]{2,90})", ans):
        cands.append(re.split(r"[。；;]", m.group(1))[0])
    m = re.search(r"(?:初步|入院|最后|最终)?诊断为\s*([^\n，,。；;]{2,40})", ans)
    if m: cands.append(m.group(1))
    m = re.search(r"初步诊断[：:]\s*([^\n]{2,60})", ans)
    if m: cands.append(m.group(1))
    out = []
    for c in cands:
        c = clean(c)
        if c.count("（") > c.count("）"): c = c[:c.rfind("（")].strip()
        if 3 <= len(c) <= 45 and not BAD_DX.search(c):
            out.append(c)
    if out:
        return out[:3]
    nums = [clean(n) for n in re.findall(r"[（(]\d+[)）]\s*([^\n。；;]{2,40})", ans)]
    nums = [n for n in nums if 3 <= len(n) <= 40 and not BAD_DX.search(n)]
    return nums[:3] if len(nums) >= 2 else []

clin = json.load(open(os.path.join(RAW, "CMB-Clin-qa.json"), encoding="utf-8"))
pool = []
for case in clin:
    for qa in case.get("QA_pairs", []):
        q = s(qa.get("question"))
        if "鉴别" in q or OPTMARK.search(q): continue
        if re.search(r"治疗|处理|原则|方案|手术", q): continue
        if not re.search(r"诊断|最可能|确诊|判断", q): continue
        kps = extract_dx(qa.get("answer"))
        if kps:
            pool.append((case, qa, kps))
print("可信诊断题池:", len(pool), "覆盖病例:", len(set(c.get("id") for c,_,_ in pool)))

ordered = sorted(pool, key=lambda t: (0 if "诊断依据" in s(t[1].get("question")) else 1, len(s(t[1].get("question")))))
sel, seen = [], set()
for case, qa, kps in ordered:
    cid = s(case.get("id"))
    if cid in seen: continue
    seen.add(cid); sel.append((case, qa, kps))
    if len(sel) >= 25: break
print("最终选中:", len(sel), "覆盖病例:", len(seen))

p = os.path.join(ROOT, "medical_diagnosis_eval.jsonl")
items = [json.loads(l) for l in open(p, encoding="utf-8")]
new_dx = []
for i, (case, qa, kps) in enumerate(sel, 1):
    new_dx.append({
        "id": f"DX3-{i:03d}", "category": "病例诊断分析", "subcategory": s(case.get("title")),
        "task": "case_diagnosis", "case_id": s(case.get("id")), "case_title": s(case.get("title")),
        "context": s(case.get("description")), "question": s(qa.get("question")),
        "reference_answer": s(qa.get("answer")), "key_points": ["诊断：" + k for k in kps],
        "key_points_note": "自动抽取，正式评分前需医学人员校核",
        "source": [x for x in items if x["id"].startswith("DX3")][0]["source"],
        "scoring": "主诊断是否命中 + 诊断依据覆盖度 + 2 名评分人 1-5 分",
        "level_hint": ["L2", "L3", "L4"], "metrics": ["准确率", "引用可追溯率", "幻觉率"],
    })

items = [it for it in items if not it["id"].startswith("DX3-")] + new_dx
items.sort(key=lambda x: x["id"])
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
print()
for it in new_dx:
    print(f"{it['id']} [{it['case_title'][:18]}] {it['question'][:26]} -> {it['key_points']}")