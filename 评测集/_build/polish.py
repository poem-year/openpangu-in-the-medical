# -*- coding: utf-8 -*-
"""收尾：规范要点文本（去换行、限长），重写三个输出文件。"""
import json, os, re, csv
ROOT = r"E:\面向医疗诊断场景的openpangu能力增强方案\评测集"

def norm(x):
    x = re.sub(r"\s+", " ", str(x or "")).strip()
    return x[:140]

p = os.path.join(ROOT, "medical_diagnosis_eval.jsonl")
items = [json.loads(l) for l in open(p, encoding="utf-8")]
for it in items:
    if it.get("key_points"):
        it["key_points"] = [norm(k) for k in it["key_points"] if norm(k)]

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
print("done", len(items))