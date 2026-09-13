# -*- coding: utf-8 -*-
"""鉴别要点抽取 v2：只在"鉴别诊断"之后的段落里找疾病名。"""
import json, os, re, csv

ROOT = r"E:\面向医疗诊断场景的openpangu能力增强方案\评测集"
def s(x): return x if isinstance(x, str) else ("" if x is None else str(x))
BAD = re.compile(r"可见|可鉴|可以|应|需|会|出现|表现|提示|不符|不能|包括|目前|待|是否|其中|多以|易于|罕见|多见|常|均|多伴|本例|该例|此例|本病例|病人|患者|若|则|但|等$|鉴别$|诊断$|"
                 r"诊断依据|实验室检查|体格检查|辅助检查|影像学检查|病理检查|病史|症状|体征|治疗|建议|原则|方法|标准|分型|要点")
def clean_name(x): return re.sub(r"\s+", " ", s(x)).strip(" 　-—、，,。；;：:")

def kp_ddx(ans):
    ans = s(ans).strip()
    m = re.search(r"鉴别诊断\s*[：:]?", ans)
    seg_src = ans[m.end():] if m else ans
    if len(clean_name(seg_src)) < 6:
        seg_src = ans
    names = []
    for seg in re.split(r"[①②③④⑤⑥⑦⑧⑨⑩]|(?:\d+)\s*[、.．)）]|[（(]\d+[)）]|\n", seg_src):
        seg = seg.strip()
        if not seg: continue
        head = clean_name(re.split(r"[：:，,。；;（(]", seg)[0])
        if 3 <= len(head) <= 26 and not BAD.search(head) and "鉴别" not in head:
            names.append(head)
    if len(names) < 2:
        m2 = re.search(r"鉴别(?:诊断)?[：:]?\s*([^\n。]{4,240})", ans)
        if m2:
            cand = [clean_name(c) for c in re.split(r"[、，,；;]", m2.group(1))]
            cand = [c for c in cand if 3 <= len(c) <= 26 and not BAD.search(c)]
            if len(cand) >= 2:
                names = cand
    out, seen = [], set()
    for n in names:
        k = re.sub(r"\s", "", n)
        if k not in seen:
            seen.add(k); out.append(n)
    return ["鉴别：" + n for n in out[:8]]

p = os.path.join(ROOT, "medical_diagnosis_eval.jsonl")
items = [json.loads(l) for l in open(p, encoding="utf-8")]
empty, noisy = [], []
for it in items:
    if it["task"] != "differential_diagnosis": continue
    kp = kp_ddx(it["reference_answer"])
    it["key_points"] = kp
    it["key_points_note"] = "自动抽取，正式评分前需医学人员校核" if kp else "自动抽取失败（参考答案为鉴别思路描述而非疾病清单），需医学人员手工补录"
    if not kp: empty.append(it["id"])
    elif len(kp) < 2: noisy.append(it["id"])

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
print("要点为空:", len(empty), empty)
print("要点少于2条的:", noisy)
print()
for it in [x for x in items if x["task"] == "differential_diagnosis"]:
    print(f"{it['id']} [{it['case_title'][:16]}] {it['question'][:22]}")
    print(f"    {it['key_points']}")