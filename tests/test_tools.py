"""工具层单元测试（无模型、无网络）。

对应《智能体设计方案.md》§6.2-A：正常路径 / 边界值 / 错误输入。
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

import agent.retrieval as retrieval_module
from agent.recorder import TurnRecorder, reset_recorder, set_recorder
from agent.retrieval import retrieve_evidence
from agent.schemas import Evidence
from agent.tools.bmi import calculate_bmi
from agent.tools.red_flags import check_red_flags, scan_input
from agent.tools.timeline import timeline_calc
from agent.tools.units import convert_units


class TestBMI:
    """calculate_bmi：中国成人标准分级。"""

    def test_normal_range(self):
        result = calculate_bmi(70, 175)
        assert "BMI = 22.9" in result
        assert "分级：正常" in result

    @pytest.mark.parametrize(
        ("weight_kg", "height_cm", "category"),
        [
            (45, 170, "偏瘦"),
            (74, 200, "正常"),   # BMI 恰好 18.5
            (96, 200, "超重"),   # BMI 恰好 24.0
            (112, 200, "肥胖"),  # BMI 恰好 28.0
        ],
    )
    def test_category_boundaries(self, weight_kg, height_cm, category):
        assert f"分级：{category}" in calculate_bmi(weight_kg, height_cm)

    def test_weight_out_of_range(self):
        assert "错误" in calculate_bmi(600, 170)
        assert "错误" in calculate_bmi(0.5, 170)

    def test_height_out_of_range(self):
        assert "错误" in calculate_bmi(70, 90)
        assert "错误" in calculate_bmi(70, 300)

    def test_non_numeric_input(self):
        assert "错误" in calculate_bmi("七十", 170)


class TestUnits:
    """convert_units：常用换算 / 拒绝剂量换算 / 错误输入。"""

    def test_jin_to_kg(self):
        assert "0.5 kg" in convert_units(1, "斤", "kg")

    def test_cm_to_inch(self):
        assert "1 英寸" in convert_units(2.54, "厘米", "英寸")

    def test_celsius_to_fahrenheit(self):
        assert "98.6 °F" in convert_units(37, "摄氏度", "华氏度")

    def test_fahrenheit_to_celsius(self):
        assert "37.0 °C" in convert_units(98.6, "f", "c")

    def test_mass_volume_conversion_refused(self):
        result = convert_units(1, "g", "ml")
        assert "错误" in result
        assert "剂量" in result

    def test_mg_ml_unsupported(self):
        assert "错误" in convert_units(500, "mg", "ml")

    def test_unsupported_pair(self):
        result = convert_units(1, "斤", "厘米")
        assert "错误" in result
        assert "支持" in result

    def test_non_numeric_input(self):
        assert "错误" in convert_units("五", "kg", "g")


class TestTimeline:
    """timeline_calc：相对/绝对时间解析与容错。"""

    def test_relative_time(self):
        result = timeline_calc(["3天前开始发热", "昨天咳嗽加重"])
        assert "时间线" in result
        assert "约3天前" in result
        assert "约1天前" in result

    def test_absolute_date(self):
        five_days_ago = (date.today() - timedelta(days=5)).isoformat()
        result = timeline_calc([f"{five_days_ago} 开始头痛"])
        assert "约5天前" in result

    def test_partially_unparsed(self):
        result = timeline_calc(["3天前发热", "前段时间不舒服"])
        assert "无法解析" in result
        assert "前段时间不舒服" in result

    def test_all_unparsed(self):
        assert "无法解析" in timeline_calc(["很久以前"])

    def test_empty_list(self):
        assert "错误" in timeline_calc([])


class TestRedFlagTool:
    """check_red_flags：急症/尽快就医/未命中/空输入。"""

    def test_emergency_hit(self):
        result = check_red_flags(["胸口压榨性疼", "大汗"])
        assert "立即急诊" in result

    def test_high_level_hit(self):
        result = check_red_flags(["小腿肿胀疼痛"])
        assert "尽快就医" in result

    def test_no_hit_is_not_safe(self):
        result = check_red_flags(["打喷嚏", "鼻塞"])
        assert "未命中" in result
        assert "不代表安全" in result

    def test_empty_input(self):
        assert "错误" in check_red_flags([])


class TestPrescan:
    """scan_input：前置安全扫描与否定保护。"""

    def test_emergency_hit(self):
        hit = scan_input("胸口压榨性疼了20分钟，还一直出汗")
        assert hit is not None
        assert hit.kind == "emergency"
        assert hit.rule_id == "chest_pain"

    def test_negation_protected(self):
        assert scan_input("现在没有压榨感了") is None

    def test_self_harm_hit(self):
        hit = scan_input("我觉得活着没意思")
        assert hit is not None
        assert hit.kind == "self_harm"

    def test_self_harm_negation_protected(self):
        assert scan_input("我不再想死了") is None

    def test_child_only_on_first_turn(self):
        hit = scan_input("孩子发烧了")
        assert hit is not None
        assert hit.kind == "child"
        assert scan_input("孩子发烧了", first_turn=False) is None

    def test_psych_only_on_first_turn(self):
        hit = scan_input("我想问下抑郁症是怎么回事")
        assert hit is not None
        assert hit.kind == "psych"

    def test_plain_input_passes(self):
        assert scan_input("最近有点鼻塞，想咨询一下") is None


class TestRetrievalStub:
    """retrieval：mock 存根行为与异常降级契约。"""

    def test_mock_returns_empty_list(self):
        assert retrieval_module.retrieve("任何查询") == []

    def test_empty_result_message(self):
        result = retrieve_evidence("咳嗽持续的原因", 3)
        assert "未找到" in result
        assert "不要编造" in result

    def test_exception_degrades_without_raising(self, monkeypatch):
        def boom(query, k=5):
            raise RuntimeError("服务不可用")

        monkeypatch.setattr(retrieval_module, "retrieve", boom)
        result = retrieve_evidence("咳嗽", 3)
        assert "不可用" in result
        assert "RuntimeError" in result

    def test_success_records_chunk_ids(self, monkeypatch):
        def fake(query, k=5):
            return [
                Evidence(
                    chunk_id="kb-001",
                    text="咳嗽持续 3 周以上建议就医评估。",
                    source="指南",
                    section="呼吸",
                    score=0.9,
                )
            ]

        monkeypatch.setattr(retrieval_module, "retrieve", fake)
        recorder = TurnRecorder()
        token = set_recorder(recorder)
        try:
            result = retrieve_evidence("咳嗽三周", 5)
        finally:
            reset_recorder(token)
        assert "kb-001" in result
        assert recorder.retrieved_chunk_ids == frozenset({"kb-001"})
        assert recorder.used_tools == ["retrieve_evidence"]
