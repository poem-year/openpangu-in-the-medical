"""思考限制与预算：不许无限思考，但也不能把正常推理掐掉。

背景：接入知识库后慢思考开始重复打转，312 题里 168 题跑满输出预算、
答案根本没写出来（见 `L2L3检索层评测报告.md` §四）。这里的规则是：
- 思考块内超过 think_budget → 掐断，交给快思考补答；
- 出现 24-gram 重复（贪心解码进循环的典型特征）→ 掐断，同样补答；
- 正常推理（没重复、没超预算）→ 绝不动它。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "eval"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pangu_infer import stop_reason_for  # noqa: E402
from run_eval import (  # noqa: E402
    RAG_BUDGET_BONUS,
    think_budget_for,
    token_budget,
)

THINK_END = 45982


class TestStopReason:
    def test_normal_reasoning_is_not_stopped(self):
        tokens = list(range(1000))
        assert stop_reason_for(tokens, think_end_id=THINK_END, think_budget=1536) is None

    def test_think_budget_caps_endless_thinking(self):
        tokens = list(range(1536))
        assert stop_reason_for(tokens, think_end_id=THINK_END, think_budget=1536) == "think_cap"

    def test_finished_thinking_is_never_capped(self):
        """已经输出 think_end 的序列，思考阶段结束，不该被 think_budget 掐。"""
        tokens = list(range(500)) + [THINK_END] + list(range(2000, 2600))
        assert stop_reason_for(tokens, think_end_id=THINK_END, think_budget=512) is None

    def test_immediate_repetition_is_detected(self):
        block = [11, 22, 33, 44, 55, 66, 77, 88] * 3  # 24 个 token 的循环片段
        tokens = [1, 2, 3] + block * 3
        assert stop_reason_for(tokens, think_end_id=THINK_END, repeat_ngram=24) == "repeat"

    def test_triple_repetition_within_window_is_detected(self):
        block = [11, 22, 33, 44, 55, 66, 77, 88] * 3
        tokens = [1, 2, 3] + block + [999] + block + [998] + block
        assert stop_reason_for(tokens, think_end_id=THINK_END, repeat_ngram=24) == "repeat"

    def test_single_repeat_is_not_a_loop(self):
        """列表/表格里 24 个 token 撞一次是常态，不能判成打转（实测误伤过）。"""
        block = [11, 22, 33, 44, 55, 66, 77, 88] * 3
        tokens = [1, 2, 3] + block + list(range(100, 200)) + block + list(range(300, 400))
        assert stop_reason_for(tokens, think_end_id=THINK_END, repeat_ngram=24) is None

    def test_repetition_after_thinking_ends_is_ignored(self):
        """答案段（出现 think_end 之后）本来就重复，一律不掐。"""
        block = [11, 22, 33, 44, 55, 66, 77, 88] * 3
        tokens = [THINK_END] + block * 4
        assert stop_reason_for(tokens, think_end_id=THINK_END, repeat_ngram=24) is None

    def test_repetition_detection_off_by_default(self):
        block = [11, 22, 33, 44, 55, 66, 77, 88] * 3
        tokens = [1, 2, 3] + block * 3
        assert stop_reason_for(tokens, think_end_id=THINK_END, repeat_ngram=0) is None

    def test_short_sequence_never_repeats(self):
        assert stop_reason_for([1, 2, 3], think_end_id=THINK_END, repeat_ngram=24) is None

    def test_think_cap_wins_when_both_trigger(self):
        tokens = [7] * 600
        reason = stop_reason_for(
            tokens, think_end_id=THINK_END, think_budget=512, repeat_ngram=24
        )
        assert reason == "think_cap"


class TestBudgets:
    def test_rag_layer_gets_a_bounded_bonus(self):
        plain = token_budget("mcq_single", "slow", "L0")
        rag = token_budget("mcq_single", "slow", "L3")
        assert rag == plain + RAG_BUDGET_BONUS
        assert rag - plain < 1000, "预算只能适度放宽，不能放开"

    def test_fast_thinking_gets_less(self):
        assert token_budget("mcq_single", "fast", "L0") < token_budget("mcq_single", "slow", "L0")

    def test_think_budget_is_half_of_output_budget(self):
        assert think_budget_for(2048, "slow") == 1024
        assert think_budget_for(320, "slow") == 512  # 保底

    def test_fast_thinking_has_no_thinking_block(self):
        assert think_budget_for(2048, "fast") is None
