# -*- coding: utf-8 -*-
"""openPangu-R-7B-2512 生成速度基准（昇腾 910B2C，单卡 bf16，无 vLLM）。

目的：把「prefill（读输入）」「decode（写输出）」分开量，避免用固定开销拉低的均值下结论。
做法：同一问题跑不同 max_new_tokens，用两点差分求纯 decode 速率（token 数之差 ÷ 时间之差）。

用法：
  source /data/openpangu/scripts/pangu_env.sh
  /data/openpangu/.venv-pangu/bin/python scripts/bench_pangu.py
"""

from __future__ import annotations

import argparse
import time

from pangu_infer import PanguModel

SHORT_QUESTION = "患者男，58岁，腰痛伴腰部活动受限，直腿抬高试验阳性，最可能的诊断是什么？"

# 一段接近真实病例的较长输入，用来量 prefill 开销
LONG_CONTEXT = (
    "患者，女，67岁，因反复活动后气促3年、加重伴双下肢水肿1周入院。"
    "既往高血压病史15年，最高血压180/100mmHg，规律服用氨氯地平；2型糖尿病病史10年，二甲双胍控制。"
    "查体：体温36.5℃，脉搏96次/分，呼吸22次/分，血压138/84mmHg。"
    "颈静脉怒张，双肺底可闻及细湿啰音，心界向左下扩大，心率96次/分，律齐，"
    "心尖区可闻及3/6级收缩期吹风样杂音，肝肋下2指，双下肢中度凹陷性水肿。"
    "辅助检查：NT-proBNP 4820pg/ml，肌钙蛋白I正常，心电图示左心室肥厚伴劳损，"
    "超声心动图示左心室舒张末内径58mm，左室射血分数38%，左房增大。"
    "问题：该患者最可能的诊断是什么？需要与哪些疾病鉴别？"
)


def run_once(model: PanguModel, question: str, max_new_tokens: int) -> dict:
    return model.chat(question, fast_thinking=True, max_new_tokens=max_new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 512, 1024])
    args = parser.parse_args()

    model = PanguModel()
    t0 = time.time()
    model.load()
    print(f"[load] {time.time() - t0:.1f}s")

    print("\n[warmup] 首次调用含算子编译，单独计时")
    warm = run_once(model, SHORT_QUESTION, 32)
    print(f"  warmup {warm['output_tokens']} tokens，{warm['elapsed_seconds']}s")

    print(f"\n[短输入 · 快思考] 输入 {warm['input_tokens']} tokens")
    rows = []
    for n in args.sizes:
        r = run_once(model, SHORT_QUESTION, n)
        rows.append((n, r["output_tokens"], r["elapsed_seconds"]))
        print(f"  max_new={n:5d} → 实际 {r['output_tokens']:5d} tokens，"
              f"{r['elapsed_seconds']:7.2f}s，均值 {r['tokens_per_second']:6.1f} tok/s")

    if len(rows) >= 2:
        (n1, t1, s1), (n2, t2, s2) = rows[0], rows[-1]
        marginal = (t2 - t1) / max(s2 - s1, 1e-6)
        print(f"  两点差分（{n1}→{n2}）：纯 decode ≈ {marginal:.1f} tok/s")

    print("\n[长输入 · 快思考] 约 300+ tokens 病例描述")
    r = run_once(model, LONG_CONTEXT, 512)
    print(f"  输入 {r['input_tokens']} tokens，输出 {r['output_tokens']} tokens，"
          f"{r['elapsed_seconds']}s，均值 {r['tokens_per_second']} tok/s")

    print("\n[短输入 · 慢思考] 会先写思考过程，输出更长")
    start = time.time()
    out = model.chat(SHORT_QUESTION, fast_thinking=False, max_new_tokens=512)
    print(f"  输入 {out['input_tokens']} tokens，输出 {out['output_tokens']} tokens，"
          f"{out['elapsed_seconds']}s，均值 {out['tokens_per_second']} tok/s，"
          f"总墙钟 {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
