# -*- coding: utf-8 -*-
import json, io
from collections import Counter

RAW = r'E:\面向医疗诊断场景的openpangu能力增强方案\评测集\_raw'
out = io.StringIO()

d = json.load(open(RAW + r'\CMB\CMB-Exam\CMB-val\CMB-val-merge.json', encoding='utf-8'))

print('== CMB val exam_class distribution ==', file=out)
for k, v in Counter(x.get('exam_class') for x in d).most_common():
    print(f'  {k}: {v}', file=out)

print('== CMB val exam_subject distribution ==', file=out)
for k, v in Counter(x.get('exam_subject') for x in d).most_common():
    print(f'  {k}: {v}', file=out)

print('== answer length ==', file=out)
print(Counter(len(x.get('answer', '')) for x in d), file=out)

clin = [x for x in d if x.get('exam_class') in ('临床医学', '医师考试', '医学考研')]
print(f'== clinical-ish count: {len(clin)} ==', file=out)
for x in clin:
    print(f"  [{x['exam_class']}|{x['exam_subject']}] {x['question'][:70]} => {x['answer']}  ({x['question_type']})", file=out)

open(RAW + r'\_inspect.txt', 'w', encoding='utf-8').write(out.getvalue())
print('written OK')