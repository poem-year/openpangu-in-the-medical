"""长期测评框架的骨架：run id 唯一、目录不覆盖、报告不覆盖、登记表只增。"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from harness import store  # noqa: E402
from harness.config import ExperimentConfig, available_configs, load_config  # noqa: E402


class TestRunId:
    def test_same_config_same_fingerprint_different_time(self):
        payload = {"engine": "direct", "layer": "L0"}
        first = store.new_run_id("t", payload, now=datetime(2026, 9, 14, 1, 0, tzinfo=timezone.utc))
        second = store.new_run_id("t", payload, now=datetime(2026, 9, 14, 2, 0, tzinfo=timezone.utc))
        assert first != second, "重复测试必须得到不同的 run id"
        assert first.split("_")[2] == second.split("_")[2], "同配置指纹应一致"

    def test_different_config_different_fingerprint(self):
        stamp = datetime(2026, 9, 14, 1, 0, tzinfo=timezone.utc)
        a = store.new_run_id("t", {"budget_scale": 1.0}, now=stamp)
        b = store.new_run_id("t", {"budget_scale": 4.0}, now=stamp)
        assert a != b


class TestCreateRun:
    def test_refuses_to_overwrite(self, tmp_path):
        run_id = "20260914T000000_x_deadbeef"
        store.create_run(run_id, {"config": {}}, runs_dir=tmp_path)
        with pytest.raises(store.RunExistsError):
            store.create_run(run_id, {"config": {}}, runs_dir=tmp_path)

    def test_writes_meta(self, tmp_path):
        run_id = "20260914T000000_x_deadbeef"
        store.create_run(run_id, {"config": {"layer": "L0"}, "params": "p"}, runs_dir=tmp_path)
        meta = json.loads((tmp_path / run_id / "meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "running" and meta["config"]["layer"] == "L0"
        store.finish_run(tmp_path / run_id, status="done")
        assert store.read_meta(tmp_path / run_id)["status"] == "done"


class TestArchiveAndIndex:
    def test_archive_does_not_overwrite_existing_report(self, tmp_path):
        run_id = "20260914T000000_x_deadbeef"
        run_dir = tmp_path / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "summary.md").write_text("新报告", encoding="utf-8")
        reports = tmp_path / "reports"
        (reports / run_id).mkdir(parents=True)
        (reports / run_id / "summary.md").write_text("旧报告", encoding="utf-8")

        store.archive_report(run_id, runs_dir=tmp_path / "runs", reports_dir=reports)
        assert (reports / run_id / "summary.md").read_text(encoding="utf-8") == "旧报告"

    def test_index_is_append_only(self, tmp_path):
        index = tmp_path / "INDEX.md"
        store.append_index({"run_id": "a", "accuracy": "10.0%"}, index_path=index)
        store.append_index({"run_id": "b", "accuracy": "20.0%"}, index_path=index)
        text = index.read_text(encoding="utf-8")
        assert text.count("| a ") == 1 and text.count("| b ") == 1
        assert text.index("| a ") < text.index("| b ")


class TestConfigs:
    def test_builtin_configs_are_valid(self):
        paths = available_configs()
        assert paths, "至少要有一个实验配置"
        for path in paths:
            cfg = load_config(path)
            assert cfg.name and cfg.engine
            if cfg.engine == "agent-batch":
                assert cfg.layer == "AGENT"

    def test_unknown_field_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yml"
        bad.write_text("name: x\nengine: direct\nbogus: 1\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(bad)

    def test_bad_engine_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.yml"
        bad.write_text("name: x\nengine: magic\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(bad)
