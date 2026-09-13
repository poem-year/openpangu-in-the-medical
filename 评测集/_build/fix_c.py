# -*- coding: utf-8 -*-
"""优化 CMB-Clin 病例题的要点抽取，并补充 case_id。"""
import json, os, re, csv

ROOT = r"E:\面向医疗诊断场景的openpangu能力增强方案\评测集"
RAW = os.path.join(ROOT, "_raw")
def s(x): return x if isinstance(x, str) else ("" if x is None else str(x))

def clean_name(x):
    x = s(x).strip()
    if x.count("（") > x.count("）"):
        x = x[:x.rfind("（")]
    if x.count("(") > x.count(")"):
        x = x[:x.rfind("(")]
    return x.strip(" 　-—:：")

def key_points_from(question, answer):
    q, ans = s(question), s(answer).strip()
    kps = []
    opts = re.findall(r"(?:^|\n)\s*([A-E])[.、．]\s*([^\n]{2,40})", q)
    m = re.search(r"答案是\s*([A-E]+)", ans)
    if opts and m:
        omap = dict(opts)
        return ["答案：" + "、".join(f"{c}.{omap.get(c, '')}" for c in m.group(1))]
    m = re.search(r"诊断[：:]\s*([^\n]{2,90})", ans)
    if m:
        diag = clean_name(re.split(r"[。；;]", m.group(1))[0])
        if diag:
            kps.append("诊断：" + diag)
    if "鉴别" in q:
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
        kps += ["鉴别：" + clean_name(n) for n in names[:6] if clean_name(n)]
    if not kps:
        head = ans.split("\n")[0].strip()
        if len(head) < 8:
            head = ans[:120]
        head = clean_name(head)[:120]
        if head:
            kps.append(head)
    return [k for k in kps if k]

# 建立 case 索引：question + answer 前60字 -> case
clin = json.load(open(os.path.join(ROOT, "_raw", "CMB-Clin-qa.json"), encoding="utf-8"))
index = {}
for case in clin:
    for qa in case.get("QA_pairs", []):
        index[(s(qa.get("question")).strip(), s(qa.get("answer")).strip()[:60])] = case
    for qa in case.get("QA_pairs", []):
        index.setdefault((s(qa.get("question")).strip(), ""), case)

p = os.path.join(ROOT, "medical_diagnosis_eval.jsonl")
items = [json.loads(l) for l in open(p, encoding="utf-8")]
fixed = 0
for it in items:
    if not it["id"].startswith("MD-C"):
        continue
    new_kp = key_points_from(it["question"], it["reference_answer"])
    if new_kp != it["key_points"]:
        fixed += 1
    it["key_points"] = new_kp
    case = index.get((it["question"].strip(), s(it["reference_answer"]).strip()[:60]))
    if case:
        it["case_id"] = s(case.get("id"))
        it["case_title"] = s(case.get("title"))

with open(p, "w", encoding="utf-8") as f:
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
        ans = it.get("answer") or "；".join(it.get("key_points") or []) or it.get("reference_answer", "")[:80]
        w.writerow([it["id"], it["category"], it["subcategory"], ans, it["scoring"], it["source"]["dataset"]])

print("修正要点题数:", fixed, "总题数:", len(items))
print("要点为空的题:", [i["id"] for i in items if not i.get("key_points") and not i.get("answer")])
for it in [x for x in items if x["id"] in ("MD-C-004", "MD-C-008", "MD-C-009", "MD-C-020", "MD-C-030")]:
    print(it["id"], "|", it["question"][:26], "|", it["key_points"])