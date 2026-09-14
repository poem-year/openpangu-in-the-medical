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

# 重复检测默认值：尾部 24 个 token 组成的片段，在最近 512 个 token 里出现过
# 就算退化打转。实测「…448.8880 就是 448.8880…」这类循环在 24-gram 上一定重复，
# 而正常推理（引用、换行、数字）几乎不会。
DEFAULT_REPEAT_NGRAM = 24
DEFAULT_REPEAT_WINDOW = 512


def stop_reason_for(
    tokens: list[int],
    *,
    think_end_id: int,
    think_budget: int | None = None,
    repeat_ngram: int = 0,
    repeat_window: int = DEFAULT_REPEAT_WINDOW,
) -> str | None:
    """纯函数：判断这一行是否该提前停止（None=继续），返回停止原因。

    两种停止原因：
    - `think_cap`：还在思考块内（没出现 think_end），且思考已用掉 `think_budget`；
    - `repeat`：还在思考块内，且尾部 `repeat_ngram` 个 token 紧挨着重复（上一段一模一样），
      或在最近 `repeat_window` 个 token 里重复出现 ≥3 次——贪心解码一旦进循环就出不来。

    调用方拿到原因后应改用快思考补答一次，而不是把思考半截当回答（见 run_eval 的救援轮）。

    **为什么只在思考块内判重复**：正式答案里本来就会有列表、表格、同类句式
    （「- 诊断标准：…」「SBP ≥140 mmHg」），24-gram 撞一次是常态；早期版本
    在答案段也判，结果把一份写得正好的答案判成打转丢掉（2026-09-14 实测踩到）。
    """
    in_thinking = think_end_id not in tokens
    if think_budget is not None and in_thinking and len(tokens) >= think_budget:
        return "think_cap"
    if repeat_ngram > 0 and in_thinking and len(tokens) >= repeat_ngram * 3:
        window = tokens[-repeat_window:] if repeat_window > 0 else tokens
        tail = tuple(window[-repeat_ngram:])
        if len(window) >= repeat_ngram * 2:
            previous = tuple(window[-repeat_ngram * 2:-repeat_ngram])
            if previous == tail:  # 紧挨着重复（逐字复读）
                return "repeat"
        occurrences = 0
        limit = len(window) - repeat_ngram + 1
        for start in range(limit):
            if tuple(window[start:start + repeat_ngram]) == tail:
                occurrences += 1
                if occurrences >= 3:
                    return "repeat"
    return None


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

    def chat_messages(
        self,
        messages: list[dict],
        *,
        fast_thinking: bool = True,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.8,
    ) -> dict:
        """按给定消息列表生成一轮（多轮对话 / 工具调用服务用）。

        messages 的 role 支持 system / user / assistant / tool，按时间顺序排列。
        与 chat() 的区别：不限制「最后一条必须是 user」，因此可以喂入
        「…工具：结果」这种以 tool 结尾的消息（openPangu 的 chat template 支持工具角色）。

        快思考：在最后一条 user/tool 消息末尾追加 " /no_think"（模型自带的开关）。
        """
        import torch

        self.load()
        payload = [dict(item) for item in messages]
        if fast_thinking:
            for item in reversed(payload):
                if item.get("role") in ("user", "tool"):
                    text = str(item.get("content") or "")
                    if not text.rstrip().endswith("/no_think"):
                        item["content"] = f"{text} /no_think"
                    break

        text = self._tokenizer.apply_chat_template(
            payload, tokenize=False, add_generation_prompt=True
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
        think_budget: int | None = None,
        repeat_ngram: int = DEFAULT_REPEAT_NGRAM,
        repeat_window: int = DEFAULT_REPEAT_WINDOW,
    ) -> dict:
        """一次跑多条问题（左填充 + 并行解码）。

        为什么用它：单条解码受权重读取带宽限制（实测约 37 tok/s），批处理能把
        整卡吞吐提到十几倍（实测 batch32 ≈ 754 tok/s，batch128 ≈ 1796 tok/s）。
        离线评测请用这个接口，不要循环调 chat()。

        返回 {"results": [...同 chat() 的 dict...], "elapsed_seconds": 总耗时,
              "total_output_tokens": 总输出 token, "tokens_per_second": 总吞吐}

        think_budget / repeat_ngram：见 `stop_reason_for`。触发时**只结束触发的那一行**
        （逐行强制 EOS，同批其它行继续），并在结果的 `stop_reason` 里标注
        （natural / think_cap / repeat / max_tokens），供上层决定是否用快思考补答。
        """
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

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
        think_end_id = tokenizer.convert_tokens_to_ids(THINK_END)
        eos_id = int(EOS_TOKEN_ID)

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

        # 逐行停止：某一行的思考超标或开始打转时，把它的 EOS logit 顶到最大，
        # 这一行当步结束，同批其它行照常继续（StoppingCriteria 做不到逐行）。
        forced_reason: dict[int, str] = {}
        if think_budget is not None or repeat_ngram > 0:
            prompt_len_local = prompt_len

            class _RowStop(LogitsProcessor):
                def __call__(self, input_ids, scores):  # noqa: D102
                    for row in range(input_ids.shape[0]):
                        if row in forced_reason:
                            continue
                        tokens = input_ids[row, prompt_len_local:].tolist()
                        reason = stop_reason_for(
                            tokens,
                            think_end_id=think_end_id,
                            think_budget=think_budget,
                            repeat_ngram=repeat_ngram,
                            repeat_window=repeat_window,
                        )
                        if reason:
                            forced_reason[row] = reason
                            # 用「本行最大 logit + 100」而不是 inf：贪心一定选中它，
                            # 万一是采样也不会把 softmax 打成 NaN。
                            scores[row, eos_id] = scores[row].max() + 100.0
                    return scores

            processors = LogitsProcessorList([_RowStop()])
        else:
            processors = None

        with torch.no_grad():
            if stopping_criteria is not None:
                gen_kwargs["stopping_criteria"] = stopping_criteria
            if processors is not None:
                gen_kwargs["logits_processor"] = processors
            outputs = self._model.generate(**gen_kwargs)
        elapsed = time.time() - start

        results = []
        total_new = 0
        for row_index, (row, question) in enumerate(zip(outputs, questions)):
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
            reason = forced_reason.get(row_index, "natural")
            if reason not in ("think_cap", "repeat") and len(new_tokens) >= max_new_tokens:
                reason = "max_tokens"
            if reason in ("think_cap", "repeat"):
                # 被掐断在思考块里：这段文本是「想了一半的话」，不是回答。
                # split_thinking 在找不到 think_end 时会把全文当 content，这里纠正过来。
                thinking, content = raw.strip(), ""
            total_new += int(new_tokens.shape[0])
            results.append(
                {
                    "content": content,
                    "thinking": thinking,
                    "raw": raw,
                    "output_tokens": int(new_tokens.shape[0]),
                    "fast_thinking": fast_thinking,
                    "stop_reason": reason,
                    "answered": bool(content.strip()),
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
