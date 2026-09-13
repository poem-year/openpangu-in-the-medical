"""数据结构校验测试（pydantic 层：字段、枚举、约束）。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schemas import (
    AgentResult,
    AgentTurnOutput,
    Assessment,
    CaseCard,
    Evidence,
    EvidenceItem,
    Hypothesis,
)


def make_assessment(**overrides) -> Assessment:
    values = {"next_steps": ["继续观察，如加重请及时就医"], "confidence": "low"}
    values.update(overrides)
    return Assessment(**values)


class TestAgentTurnOutput:
    def test_missing_reply_rejected(self):
        with pytest.raises(ValidationError):
            AgentTurnOutput(
                assessment=make_assessment(),
                case_card=CaseCard(),
                risk_level="low",
            )

    def test_invalid_risk_level_rejected(self):
        with pytest.raises(ValidationError):
            AgentTurnOutput(
                reply="你好",
                assessment=make_assessment(),
                case_card=CaseCard(),
                risk_level="critical",
            )

    def test_valid_output_with_defaults(self):
        output = AgentTurnOutput(
            reply="你好",
            assessment=make_assessment(),
            case_card=CaseCard(),
            risk_level="low",
        )
        assert output.case_card.age_group == "未提供"
        assert output.case_card.symptoms == []
        assert output.assessment.hypotheses == []


class TestAssessment:
    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValidationError):
            make_assessment(confidence="very_high")

    def test_next_steps_must_not_be_empty(self):
        with pytest.raises(ValidationError):
            make_assessment(next_steps=[])

    def test_defaults(self):
        assessment = make_assessment()
        assert assessment.hypotheses == []
        assert assessment.evidence == []
        assert assessment.missing_info == []


class TestHypothesis:
    def test_invalid_likelihood_rejected(self):
        with pytest.raises(ValidationError):
            Hypothesis(name="急性支气管炎", likelihood="certain")

    def test_valid(self):
        item = Hypothesis(name="急性支气管炎", likelihood="medium")
        assert item.supporting == []
        assert item.against == []


class TestEvidenceItem:
    def test_invalid_source_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceItem(source="web", ref="x", detail="y")

    def test_valid_sources(self):
        for source in ("dialogue", "tool", "kb"):
            item = EvidenceItem(source=source, ref="r", detail="d")
            assert item.source == source


class TestAgentResult:
    def test_used_tools_defaults_to_empty(self):
        result = AgentResult(
            reply="r",
            assessment=make_assessment(),
            risk_level="low",
            handled_as="normal",
        )
        assert result.used_tools == []

    def test_invalid_handled_as_rejected(self):
        with pytest.raises(ValidationError):
            AgentResult(
                reply="r",
                assessment=make_assessment(),
                risk_level="low",
                handled_as="unknown",
            )


class TestEvidence:
    def test_defaults(self):
        item = Evidence(chunk_id="kb-001", text="正文")
        assert item.source == ""
        assert item.section == ""
        assert item.score == 0.0
