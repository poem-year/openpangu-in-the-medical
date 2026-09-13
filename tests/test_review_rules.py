"""审查规则单元测试（纯函数，不涉及模型服务）。

每个红线模式「命中 + 不误伤」用例。
"""

from __future__ import annotations

import json

from agent.review.engine import run_review
from agent.review.rules import ReviewContext, check_output
from agent.schemas import AgentTurnOutput, Assessment, CaseCard, EvidenceItem, Hypothesis
from fake_model import ScriptedFakeChatModel, text_response


def make_output(
    reply: str = "目前考虑可能是普通上呼吸道感染，建议多休息，若加重请及时就医。",
    *,
    confidence: str = "low",
    hypotheses: list[Hypothesis] | None = None,
    evidence: list[EvidenceItem] | None = None,
    next_steps: list[str] | None = None,
    red_flags_found: list[str] | None = None,
    risk_level: str = "low",
) -> AgentTurnOutput:
    return AgentTurnOutput(
        reply=reply,
        assessment=Assessment(
            hypotheses=hypotheses or [],
            evidence=evidence or [],
            missing_info=[],
            next_steps=next_steps or ["继续观察，如症状加重请及时就医"],
            confidence=confidence,
        ),
        case_card=CaseCard(red_flags_found=red_flags_found or []),
        risk_level=risk_level,
    )


def codes(violations) -> set[str]:
    return {violation.code for violation in violations}


class TestDiagnosisLanguage:
    def test_plain_diagnosis_hit(self):
        output = make_output("您得了肺癌，需要尽快手术。")
        assert "diagnosis_language" in codes(check_output(output, ReviewContext()))

    def test_quoted_speech_is_guarded(self):
        output = make_output("您说您得了肺癌，这一点我无法判断，建议您尽快就医让医生面诊确认。")
        assert "diagnosis_language" not in codes(check_output(output, ReviewContext()))

    def test_negated_statement_is_guarded(self):
        output = make_output("现有信息不能确诊为肺炎，建议您到医院进一步检查，尽快就医。")
        assert "diagnosis_language" not in codes(check_output(output, ReviewContext()))


class TestMedication:
    def test_drug_recommendation_hit(self):
        output = make_output("建议您吃布洛芬，这个药很常见。")
        assert "medication_recommendation" in codes(check_output(output, ReviewContext()))

    def test_dose_amount_hit(self):
        output = make_output("可以每次服用2片，饭后吃。")
        assert "medication_recommendation" in codes(check_output(output, ReviewContext()))

    def test_negated_mention_not_hit(self):
        output = make_output("请不要自行服用布洛芬等药物，具体用药请咨询医生或药师。")
        assert "medication_recommendation" not in codes(check_output(output, ReviewContext()))


class TestAbsolutes:
    def test_absolute_promise_hit(self):
        output = make_output("您肯定没事，不用去医院。")
        assert "absolutist_claim" in codes(check_output(output, ReviewContext()))


class TestCitations:
    def test_fabricated_chunk_id_flagged(self):
        output = make_output(
            evidence=[EvidenceItem(source="kb", ref="chunk-999", detail="编造的引用")]
        )
        ctx = ReviewContext(retrieved_chunk_ids=frozenset({"chunk-001"}))
        violations = check_output(output, ctx)
        assert "fabricated_citation" in codes(violations)
        violation = next(v for v in violations if v.code == "fabricated_citation")
        assert violation.action == "degrade"
        assert violation.refs == ("chunk-999",)

    def test_valid_chunk_id_passes(self):
        output = make_output(
            evidence=[EvidenceItem(source="kb", ref="chunk-001", detail="真实引用")]
        )
        ctx = ReviewContext(retrieved_chunk_ids=frozenset({"chunk-001"}))
        assert "fabricated_citation" not in codes(check_output(output, ctx))

    def test_reference_without_retrieval_flagged(self):
        output = make_output("根据相关研究，这种情况大多问题不大，建议继续观察，加重请就医。")
        assert "unsupported_reference" in codes(check_output(output, ReviewContext()))

    def test_reference_allowed_when_retrieved(self):
        output = make_output(
            "根据相关资料，建议您近期就医评估。",
            evidence=[EvidenceItem(source="kb", ref="chunk-001", detail="真实引用")],
        )
        ctx = ReviewContext(retrieved_chunk_ids=frozenset({"chunk-001"}))
        assert "unsupported_reference" not in codes(check_output(output, ctx))


class TestRedFlagConsistency:
    def test_inconsistent_risk_level_flagged(self):
        output = make_output(
            reply="可能性不大，先在家观察两天。",
            red_flags_found=["胸痛"],
            risk_level="low",
        )
        assert "red_flag_escalation" in codes(check_output(output, ReviewContext()))

    def test_consistent_output_passes(self):
        output = make_output(
            reply="这种情况需要尽快就医。",
            red_flags_found=["胸痛"],
            risk_level="high",
        )
        assert "red_flag_escalation" not in codes(check_output(output, ReviewContext()))


class TestClosing:
    def test_confidence_capped_at_closing(self):
        output = make_output(confidence="high", next_steps=["尽快到医院呼吸科就诊"])
        ctx = ReviewContext(is_closing=True)
        assert "closing_confidence_cap" in codes(check_output(output, ctx))

    def test_advice_required_for_medium_confidence(self):
        output = make_output(confidence="medium", next_steps=["多喝水，注意休息"])
        assert "closing_advice_missing" in codes(check_output(output, ReviewContext()))

    def test_high_confidence_allowed_before_closing(self):
        output = make_output(
            confidence="high",
            hypotheses=[Hypothesis(name="急性支气管炎", likelihood="medium")],
            next_steps=["尽快到医院就诊"],
        )
        assert not codes(check_output(output, ReviewContext()))


class TestSelfHarmEcho:
    def test_echo_flagged(self):
        output = make_output("如果您再次出现自杀的念头，请立即告诉家人。")
        assert "self_harm_echo" in codes(check_output(output, ReviewContext()))


class TestStructure:
    def test_empty_reply_flagged(self):
        output = make_output("   ")
        assert "empty_reply" in codes(check_output(output, ReviewContext()))

    def test_empty_hypotheses_with_medium_confidence_flagged(self):
        output = make_output(confidence="medium", next_steps=["建议近期到医院就诊"])
        assert "empty_hypotheses" in codes(check_output(output, ReviewContext()))


class TestEngineActions:
    def test_replace_fallback_without_model(self):
        output = make_output("您得了肺炎，建议您服用抗生素。")
        templates = {"refuse": "SAFE_REPLY", "self_harm_suffix": "SUFFIX"}
        result = run_review(
            output,
            ReviewContext(),
            model=None,
            fix_prompt="请修正",
            templates=templates,
        )
        assert result.action == "replace"
        assert result.output.reply == "SAFE_REPLY"

    def test_fix_attempt_success(self):
        fixed = make_output(
            "目前考虑需要排除肺部感染的可能，建议尽快到医院呼吸科面诊确认。",
            confidence="medium",
            next_steps=["建议尽快到医院呼吸科就诊"],
        )
        model = ScriptedFakeChatModel(
            responses=[text_response(json.dumps(fixed.model_dump(), ensure_ascii=False))]
        )
        output = make_output("您得了肺癌，需要尽快手术。")
        result = run_review(
            output,
            ReviewContext(),
            model=model,
            fix_prompt="请修正",
            templates={},
        )
        assert result.action == "fix"
        assert "您得了" not in result.output.reply

    def test_failed_fix_falls_back_to_replace(self):
        model = ScriptedFakeChatModel(responses=[])
        output = make_output("您得了肺癌，需要尽快手术。")
        result = run_review(
            output,
            ReviewContext(),
            model=model,
            fix_prompt="请修正",
            templates={"refuse": "SAFE_REPLY"},
        )
        assert result.action == "replace"
        assert result.output.reply == "SAFE_REPLY"
