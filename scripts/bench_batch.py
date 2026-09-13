# -*- coding: utf-8 -*-
"""openPangu-R-7B-2512 批处理吞吐基准。

目的：验证「单条解码受显存带宽限制、批处理才能把吞吐提上去」。
做法：同一时刻并行跑 N 条序列，测总吞吐（所有序列输出 token 之和 ÷ 总耗时）。

用法：
  source /data/openpangu/scripts/pangu_env.sh
  /data/openpangu/.venv-pangu/bin/python scripts/bench_batch.py --batch-sizes 1 4 8 16 32
"""

from __future__ import annotations

import argparse
import time

from pangu_infer import PanguModel

QUESTIONS = [
    "患者男，58岁，腰痛伴活动受限，直腿抬高试验阳性，最可能的诊断是什么？",
    "患者65岁，桶状胸，心尖搏动在剑突下，肺动脉瓣第二心音增强，首先考虑什么病？",
    "女性32岁，甲状腺功能检查示TSH降低、FT4升高，最可能的诊断是什么？",
    "男童6岁，发热3天伴皮疹，口腔可见科氏斑，最可能的诊断是什么？",
    "患者，男，45岁，突发胸痛2小时，心电图ST段抬高，最可能的诊断是什么？",
    "女性28岁，反复关节痛伴面部蝶形红斑，抗核抗体阳性，最可能的诊断是什么？",
    "患者，男，68岁，进行性吞咽困难3个月，体重下降10kg，首先考虑什么病？",
    "女性50岁，突发右侧肢体无力伴言语不清，头颅CT未见出血，最可能的诊断是什么？",
]


def build_batch(tokenizer, prompts: list[str], system_prompt: str, device: str):
    """左填充成批输入，返回 input_ids / attention_mask（贴在 NPU 上）。"""
    import torch

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    tokenizer.padding_side = "left"
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    import torch
    from pangu_infer import DEFAULT_SYSTEM_PROMPT

    model = PanguModel()
    model.load()
    tok = model._tokenizer
    net = model._model

    # 预热：算子编译只发生一次
    with torch.no_grad():
        ids, mask = build_batch(tok, QUESTIONS[:1], DEFAULT_SYSTEM_PROMPT, model.device)
        net.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8,
                     do_sample=False, eos_token_id=45892, pad_token_id=0)
    print("[warmup] 完成\n")
    print(f"{'batch':>5} {'输出token':>9} {'耗时s':>8} {'总吞吐':>10} {'单条吞吐':>10}")

    for bs in args.batch_sizes:
        prompts = [QUESTIONS[i % len(QUESTIONS)] for i in range(bs)]
        ids, mask = build_batch(tok, prompts, DEFAULT_SYSTEM_PROMPT, model.device)
        start = time.time()
        with torch.no_grad():
            out = net.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=45892,
                pad_token_id=0,
            )
        elapsed = time.time() - start
        produced = int((out[:, ids.shape[1]:] != 0).sum())  # 粗略统计非 padding 输出
        total = produced / elapsed
        per_seq = total / bs
        print(f"{bs:>5} {produced:>9} {elapsed:>8.2f} {total:>8.1f} t/s {per_seq:>8.1f} t/s")


if __name__ == "__main__":
    main()
