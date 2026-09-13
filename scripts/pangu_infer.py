# -*- coding: utf-8 -*-
"""openPangu-R-7B-2512 本地推理入口（昇腾 910B2C / CANN 8.3.RC1 / torch-npu 2.7.1）。

两种用法：

1) 命令行单轮问答

   source /data/openpangu/scripts/pangu_env.sh
   /data/openpangu/.venv-pangu/bin/python scripts/pangu_infer.py "患者男，58岁，腰痛……"
   # 快思考（不输出思考过程）：加 --fast；要 JSON：加 --json

2) 当模块导入（评测脚本、后续服务化都用这个）

   from scripts.pangu_infer import PanguModel
   llm = PanguModel()
   out = llm.chat("58岁男性腰痛，直腿抬高试验阳性，最可能诊断？")
   print(out["content"])

说明：
- 默认慢思考（模型自带能力），在用户输入末尾追加 " /no_think" 可切到快思考。
- 评测要求可复现：默认贪心解码（do_sample=False）。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

MODEL_PATH = "/data/models/openPangu-R-7B-2512"
DEVICE = "npu:0"
EOS_TOKEN_ID = 45892

DEFAULT_SYSTEM_PROMPT = (
    "你必须严格遵守法律法规和社会道德规范。"
    "生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。"
)

THINK_START = "[unused16]"
THINK_END = "[unused17]"
TEXT_END = "[unused10]"


def split_thinking(text: str) -> tuple[str, str]:
    """把模型原始输出拆成 (thinking, content)。快思考时 thinking 为空串。"""
    if THINK_END in text:
        thinking = text.split(THINK_END)[0].split(THINK_START)[-1].strip()
        content = text.split(THINK_END)[-1].split(TEXT_END)[0].strip()
        return thinking, content
    return "", text.split(TEXT_END)[0].strip()


@dataclass
class PanguModel:
    model_path: str = MODEL_PATH
    device: str = DEVICE
    _model: object = field(default=None, init=False, repr=False)
    _tokenizer: object = field(default=None, init=False, repr=False)

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        import torch_npu  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=False, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self._model = model.to(self.device).eval()

    def chat(
        self,
        question: str,
        *,
        history: list[dict] | None = None,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        fast_thinking: bool = False,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.8,
    ) -> dict:
        """单轮（或带历史）问答，返回 thinking / content / 统计信息。"""
        import torch

        self.load()
        prompt = question if fast_thinking else question
        if fast_thinking and not question.rstrip().endswith("/no_think"):
            prompt = question + " /no_think"

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history or [])
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=0,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        start = time.time()
        with torch.no_grad():
            outputs = self._model.generate(**gen_kwargs)
        elapsed = time.time() - start

        new_tokens = outputs[0][input_ids.shape[1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=False)
        thinking, content = split_thinking(raw)
        return {
            "content": content,
            "thinking": thinking,
            "raw": raw,
            "input_tokens": int(input_ids.shape[1]),
            "output_tokens": int(new_tokens.shape[0]),
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(new_tokens.shape[0] / max(elapsed, 1e-6), 1),
            "fast_thinking": fast_thinking,
        }

    def chat_batch(
        self,
        questions: list[str],
        *,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        fast_thinking: bool = False,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.8,
        progress_cb=None,
    ) -> dict:
        """一次跑多条问题（左填充 + 并行解码）。

        为什么用它：单条解码受权重读取带宽限制（实测约 37 tok/s），批处理能把
        整卡吞吐提到十几倍（实测 batch32 ≈ 754 tok/s，batch128 ≈ 1796 tok/s）。
        离线评测请用这个接口，不要循环调 chat()。

        返回 {"results": [...同 chat() 的 dict...], "elapsed_seconds": 总耗时,
              "total_output_tokens": 总输出 token, "tokens_per_second": 总吞吐}
        """
        import torch

        if not questions:
            return {"results": [], "elapsed_seconds": 0.0, "total_output_tokens": 0,
                    "tokens_per_second": 0.0}

        self.load()
        tokenizer = self._tokenizer
        texts = []
        for question in questions:
            prompt = question
            if fast_thinking and not question.rstrip().endswith("/no_think"):
                prompt = question + " /no_think"
            messages: list[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        # 生成必须左填充：从右侧续写，右侧不能有 pad
        original_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            enc = tokenizer(texts, return_tensors="pt", padding=True)
        finally:
            tokenizer.padding_side = original_side

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=0,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        start = time.time()
        stopping_criteria = None
        if progress_cb is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

            prompt_len_local = prompt_len

            class _ProgressHook(StoppingCriteria):
                """每解码一步回调一次，用来给看板报"批内已产出多少 token"。"""

                def __call__(self, input_ids, scores, **kwargs):  # noqa: D102
                    progress_cb(int(input_ids.shape[1]) - prompt_len_local)
                    return False

            stopping_criteria = StoppingCriteriaList([_ProgressHook()])

        with torch.no_grad():
            if stopping_criteria is not None:
                gen_kwargs["stopping_criteria"] = stopping_criteria
            outputs = self._model.generate(**gen_kwargs)
        elapsed = time.time() - start

        results = []
        total_new = 0
        for row, question in zip(outputs, questions):
            new_tokens = row[prompt_len:]
            if EOS_TOKEN_ID in new_tokens.tolist():
                stop_at = new_tokens.tolist().index(EOS_TOKEN_ID)
                new_tokens = new_tokens[: stop_at + 1]
            # 去掉尾部 pad（pad_token_id=0，仅出现在已结束的序列末尾）
            keep = (new_tokens != 0).nonzero().flatten()
            if len(keep):
                new_tokens = new_tokens[: int(keep[-1]) + 1]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
            thinking, content = split_thinking(raw)
            total_new += int(new_tokens.shape[0])
            results.append(
                {
                    "content": content,
                    "thinking": thinking,
                    "raw": raw,
                    "output_tokens": int(new_tokens.shape[0]),
                    "fast_thinking": fast_thinking,
                }
            )

        return {
            "results": results,
            "elapsed_seconds": round(elapsed, 2),
            "total_output_tokens": total_new,
            "tokens_per_second": round(total_new / max(elapsed, 1e-6), 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="openPangu-R-7B-2512 单轮问答")
    parser.add_argument("question", help="要问的问题")
    parser.add_argument("--fast", action="store_true", help="快思考（等价于输入末尾加 /no_think）")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--json", action="store_true", help="输出 JSON（含 thinking 与统计）")
    args = parser.parse_args()

    llm = PanguModel()
    result = llm.chat(
        args.question,
        system_prompt=None if args.no_system_prompt else DEFAULT_SYSTEM_PROMPT,
        fast_thinking=args.fast,
        max_new_tokens=args.max_new_tokens,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result["content"])
    print(
        f"\n[统计] 输入 {result['input_tokens']} / 输出 {result['output_tokens']} tokens，"
        f"{result['elapsed_seconds']}s，{result['tokens_per_second']} tok/s，"
        f"模式={'快思考' if result['fast_thinking'] else '慢思考'}"
    )


if __name__ == "__main__":
    main()
