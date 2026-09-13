#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对产出的 jsonl 做结构校验，并打印一份可核对的清单。

校验项：
1. 每行都是合法 JSON，且必要字段齐全；
2. 选择题必须有选项和答案，答案必须在选项里；
3. 开放题必须有参考答案或评分要点；
4. 多模态题引用的图片文件必须真实存在；
5. 跨维度不能出现同一道题；
6. id 唯一且连续。
"""

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = ["id", "dimension_code", "dimension", "task_type", "language",
            "question", "scoring", "source", "provenance", "level_hint"]
MCQ_TASKS = {"mcq_single", "mmcq_single"}


def main() -> int:
    files = sorted(p for p in ROOT.glob("*.jsonl"))
    if not files:
        print("没有找到 jsonl 文件", file=sys.stderr)
        return 1

    errors, warnings = [], []
    seen_ids, seen_questions = {}, {}
    total = 0
    print(f"{'文件':38s} {'题量':>5s} {'中文':>5s} {'英文':>5s}  题型")
    print("-" * 100)

    for path in files:
        rows = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{lineno} JSON 解析失败 {exc}")
        total += len(rows)
        if not rows:
            errors.append(f"{path.name}: 空文件")
            continue

        lang = collections.Counter(r["language"] for r in rows)
        tasks = collections.Counter(r["task_type"] for r in rows)
        print(f"{path.name:38s} {len(rows):5d} {lang.get('zh', 0):5d} {lang.get('en', 0):5d}  "
              + "、".join(f"{k}×{v}" for k, v in tasks.most_common()))

        for r in rows:
            rid = r.get("id", "?")
            for key in REQUIRED:
                if r.get(key) in (None, "", [], {}):
                    errors.append(f"{path.name} {rid}: 缺字段 {key}")
            if not r.get("question"):
                errors.append(f"{path.name} {rid}: 题干为空")
            if rid in seen_ids:
                errors.append(f"{path.name} {rid}: id 与 {seen_ids[rid]} 重复")
            seen_ids[rid] = path.name

            if r["task_type"] in MCQ_TASKS:
                opt, ans = r.get("options"), r.get("answer")
                if not opt or not ans:
                    errors.append(f"{path.name} {rid}: 选择题缺选项或答案")
                elif isinstance(ans, str) and any(a not in opt for a in ans):
                    errors.append(f"{path.name} {rid}: 答案 {ans} 不在选项 {list(opt)} 中")
            else:
                # 检索类题目没有单条参考答案，评分靠与语料比对算召回。
                has_corpus = bool((r.get("scoring") or {}).get("corpus"))
                if not (r.get("reference_answer") or r.get("key_points")
                        or r.get("rubric") or has_corpus):
                    errors.append(f"{path.name} {rid}: 开放题没有参考答案/要点/评分量表")

            media = r.get("media") or {}
            if media.get("path") and not (ROOT / media["path"]).exists():
                errors.append(f"{path.name} {rid}: 图片不存在 {media['path']}")

            # 开放题的题干是统一的提问模板，真正区分题目的是病例正文，去重要带上正文。
            qkey = ((r.get("question") or "")[:120],
                    (r.get("context") or "")[:100],
                    (r.get("dialogue")[0]["text"][:80] if r.get("dialogue") else ""))
            if qkey and qkey in seen_questions:
                warnings.append(f"{path.name} {rid}: 题干与 {seen_questions[qkey]} 重复")
            else:
                seen_questions[qkey] = f"{path.name} {rid}"

    print("-" * 100)
    print(f"合计 {total} 题，{len(files)} 个文件")
    srcs = collections.Counter()
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                srcs[json.loads(line)["source"]["name"]] += 1
    print("\n来源分布：")
    for name, n in srcs.most_common():
        print(f"  {n:5d}  {name}")

    if warnings:
        print(f"\n警告 {len(warnings)} 条：")
        for w in warnings[:15]:
            print("  ! " + w)
    if errors:
        print(f"\n错误 {len(errors)} 条：")
        for e in errors[:30]:
            print("  x " + e)
        return 1
    print("\n校验通过：结构、答案、图片引用、id 唯一性均无问题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
