# -*- coding: utf-8 -*-
"""openPangu-R-7B-2512 本机冒烟测试（昇腾 910B2C + torch-npu，无 vLLM）。

用官方仓库的 HuggingFace 参考实现加载权重，验证三件事：
  1. 模型能在 NPU 上加载（bf16，显存是否够）；
  2. 单轮生成能出结果，token 数与耗时可见；
  3. 慢思考 / 快思考两种模式能切换（快思考＝用户输入末尾加 " /no_think"）。

用法：
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
  /data/openpangu/.venv-pangu/bin/python scripts/smoke_pangu.py
"""

from __future__ import annotations

import argparse
import time

SYSTEM_PROMPT = (
    "你必须严格遵守法律法规和社会道德规范。"
    "生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。"
)

# 该模型的输出把思考内容放在 [unused16]/[unused17] 之间，正文在 [unused17] 之后。
THINK_END = "[unused17]"
THINK_START = "[unused16]"
TEXT_END = "[unused10]"


def split_thinking(text: str) -> tuple[str, str]:
    """拆出 (thinking, content)；没有思考段时 thinking 为空。"""
    if THINK_END in text:
        thinking = text.split(THINK_END)[0].split(THINK_START)[-1].strip()
        content = text.split(THINK_END)[-1].split(TEXT_END)[0].strip()
        return thinking, content
    return "", text.split(TEXT_END)[0].strip()


def build_inputs(tokenizer, prompt: str, system_prompt: str | None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], return_tensors="pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/models/openPangu-R-7B-2512")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument(
        "--prompt",
        default="患者男，58岁，腰痛伴腰部活动受限，脊柱叩击痛，直腿抬高试验阳性，最可能的诊断是什么？",
    )
    parser.add_argument("--skip-fast", action="store_true", help="跳过快思考那一轮")
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401  导入即注册 NPU 后端
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(
        f"[env] torch={torch.__version__} torch_npu={torch_npu.__version__} "
        f"cann={getattr(torch.version, 'cann', None)}"
    )
    print(f"[env] npu_available={torch.npu.is_available()} device_count={torch.npu.device_count()}")

    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = model.to(args.device).eval()
    print(f"[load] 模型加载完成，耗时 {time.time() - load_start:.1f}s")

    system_prompt = None if args.no_system_prompt else SYSTEM_PROMPT

    runs = [("慢思考（默认）", args.prompt)]
    if not args.skip_fast:
        runs.append(("快思考（/no_think）", args.prompt + " /no_think"))

    for name, prompt in runs:
        inputs = build_inputs(tokenizer, prompt, system_prompt)
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=45892,
                pad_token_id=0,
            )
        elapsed = time.time() - start
        generated = outputs[0][input_ids.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=False)
        thinking, content = split_thinking(text)

        print("=" * 70)
        print(
            f"[{name}] 输入 {input_ids.shape[1]} tokens，输出 {generated.shape[0]} tokens，"
            f"耗时 {elapsed:.1f}s（{generated.shape[0] / max(elapsed, 1e-6):.1f} tok/s）"
        )
        if thinking:
            print(f"--- thinking（{len(thinking)} 字）---\n{thinking[:1500]}")
        print(f"--- content ---\n{content[:1500]}")


if __name__ == "__main__":
    main()
