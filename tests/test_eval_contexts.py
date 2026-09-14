"""预检索结果的完整性检查。

踩过的坑：一次被中断的 L3 预检索只写了 19 题就死了，重跑时
`ensure_contexts` 发现行数不够、去调 `stage_retrieve`，而后者见到
contexts.jsonl 已存在就直接跳过——结果 298 道题拿到的都是「未检索到资料」，
整轮实验静默失真。所以这里盯住两件事：缺题要强制重跑、重跑后仍缺要报错。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval"
for path in (str(EVAL_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

run_eval = pytest.importorskip("run_eval")


class _Args:
    layer = "L3"
    kb_k = None


def _write_contexts(run_dir: Path, ids: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "contexts.jsonl").open("w", encoding="utf-8") as fh:
        for item_id in ids:
            fh.write(json.dumps({"id": item_id, "evidence": []}, ensure_ascii=False) + "\n")


def _items(ids: list[str]) -> list[dict]:
    return [{"id": item_id, "question": "问题"} for item_id in ids]


class TestEnsureContexts:
    def test_non_rag_layer_skips_retrieval(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_eval, "layer_has_rag", lambda layer: False)
        called = []
        monkeypatch.setattr(run_eval, "stage_retrieve", lambda *a, **k: called.append(1))
        assert run_eval.ensure_contexts(_items(["A"]), _Args(), str(tmp_path)) == {}
        assert called == []

    def test_complete_contexts_are_reused(self, tmp_path, monkeypatch):
        _write_contexts(tmp_path, ["A", "B"])
        called = []
        monkeypatch.setattr(run_eval, "stage_retrieve", lambda *a, **k: called.append(1))
        contexts = run_eval.ensure_contexts(_items(["A", "B"]), _Args(), str(tmp_path))
        assert set(contexts) == {"A", "B"}
        assert called == []

    def test_partial_contexts_force_rerun(self, tmp_path, monkeypatch):
        """行数不够时必须重跑，而且要带 force——否则 stage_retrieve 会跳过。"""
        _write_contexts(tmp_path, ["A"])
        seen = {}

        def fake_retrieve(items, args, run_dir, *, force=False):
            seen["force"] = force
            _write_contexts(Path(run_dir), [item["id"] for item in items])

        monkeypatch.setattr(run_eval, "stage_retrieve", fake_retrieve)
        contexts = run_eval.ensure_contexts(_items(["A", "B"]), _Args(), str(tmp_path))
        assert seen["force"] is True
        assert set(contexts) == {"A", "B"}

    def test_still_missing_after_rerun_raises(self, tmp_path, monkeypatch):
        _write_contexts(tmp_path, ["A"])
        monkeypatch.setattr(run_eval, "stage_retrieve", lambda *a, **k: None)
        with pytest.raises(SystemExit):
            run_eval.ensure_contexts(_items(["A", "B"]), _Args(), str(tmp_path))
