"""生成阶段（stage_generate）的集成测试，用假模型替掉 NPU。

为什么必须有：2026-09-14 一次改动把 `out = model.chat_batch(...)` 的赋值行删掉了，
语法检查发现不了（只是 NameError），单测覆盖不到，结果 L3 那一轮 19 秒"跑完"、
0 道题，还自动归档推送了一份空报告。这个测试就是防这一类。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT / "eval"), str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

run_eval = pytest.importorskip("run_eval")


class _FakeModel:
    """按行返回：正常答题 / 被掐断（无 content）/ 补答成功。"""

    calls: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def load(self) -> None:
        pass

    def chat_batch(self, questions, *, system_prompt=None, fast_thinking=False,
                   max_new_tokens=1024, progress_cb=None, think_budget=None, **_):
        _FakeModel.calls.append(
            {"n": len(questions), "fast": fast_thinking, "max_new_tokens": max_new_tokens,
             "think_budget": think_budget}
        )
        results = []
        for index, _ in enumerate(questions):
            if not fast_thinking and index == 0:
                # 第一轮第一题：模拟「思考打转被掐断」
                results.append({"content": "", "thinking": "…", "raw": "…",
                                "output_tokens": 1, "fast_thinking": False,
                                "stop_reason": "repeat", "answered": False})
            else:
                results.append({"content": "答案：B", "thinking": "", "raw": "答案：B",
                                "output_tokens": 6, "fast_thinking": fast_thinking,
                                "stop_reason": "natural", "answered": True})
        return {"results": results, "elapsed_seconds": 0.1,
                "total_output_tokens": 6 * len(questions), "tokens_per_second": 60.0}


class _Args:
    layer = "L3"
    thinking = "slow"
    batch_size = 2
    budget_scale = 4.0
    budget_cap = 0
    think_ratio = 1.0
    per_dimension = None
    kb_k = None


def _items() -> list[dict]:
    return [
        {"id": "D01-0001", "dimension_code": "D01", "task_type": "mcq_single",
         "question": "题干一", "options": {"A": "甲", "B": "乙"}},
        {"id": "D01-0002", "dimension_code": "D01", "task_type": "mcq_single",
         "question": "题干二", "options": {"A": "甲", "B": "乙"}},
    ]


class TestStageGenerate:
    def test_writes_predictions_and_rescues(self, tmp_path, monkeypatch):
        _FakeModel.calls = []
        monkeypatch.setattr(run_eval, "PanguModel", _FakeModel)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # 预检索结果要覆盖全部题号，否则 ensure_contexts 会去真跑检索
        (run_dir / "contexts.jsonl").write_text(
            "".join(
                json.dumps({"id": item["id"], "evidence": [], "retrieved_chunk_ids": []},
                           ensure_ascii=False) + "\n"
                for item in _items()
            ),
            encoding="utf-8",
        )

        run_eval.stage_generate(_items(), _Args(), str(run_dir), "fp-test")

        rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2, "两题都要落盘"
        assert all(row["content"].strip() for row in rows), "被掐断的题要补答，不能留空"
        first = rows[0]
        assert first["rescued"] is True and first["rescue_reason"] == "repeat"
        assert first["truncated"] is False
        assert rows[1]["rescued"] is False

        # 预算按 4 倍放大，且默认不设思考早停线
        main_calls = [c for c in _FakeModel.calls if not c["fast"]]
        assert main_calls and main_calls[0]["max_new_tokens"] == int(
            run_eval.token_budget("mcq_single", "slow", "L3") * 4
        )
        assert main_calls[0]["think_budget"] is None
        assert any(c["fast"] for c in _FakeModel.calls), "要有一次快思考补答"
