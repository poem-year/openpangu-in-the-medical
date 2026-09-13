"""智能体模块的数据结构（字段与枚举的单一来源）。

与《智能体接口规范.md》§1 一一对应；修改任何字段或枚举，
都必须同步更新接口规范文档，并走版本变更流程。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Likelihood = Literal["low", "medium", "high"]
RiskLevel = Literal["low", "medium", "high", "emergency"]
HandledAs = Literal["normal", "emergency", "out_of_scope", "review_modified"]
EvidenceSource = Literal["dialogue", "tool", "kb"]


class Hypothesis(BaseModel):
    """一条候选假设（鉴别方向）。"""

    name: str = Field(description="假设名称，如「急性支气管炎」")
    likelihood: Likelihood = Field(description="相对可能性（不是概率）")
    supporting: list[str] = Field(default_factory=list, description="支持证据（对话/工具/检索要点）")
    against: list[str] = Field(default_factory=list, description="反对证据；写不出任何反证的假设不可轻信")
    missing: list[str] = Field(default_factory=list, description="该假设还缺的关键信息")


class EvidenceItem(BaseModel):
    """一条判断依据。"""

    source: EvidenceSource = Field(description="来源：dialogue=患者对话 / tool=工具结果 / kb=知识库检索")
    ref: str = Field(description="引用定位：对话要点或轮次 / 工具名(参数摘要) / 知识库 chunk_id（必须真实来自检索返回）")
    detail: str = Field(description="依据内容一句话")


class Assessment(BaseModel):
    """面向输出的结构化评估。"""

    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="当前鉴别方向，按可能性降序；信息不足时可为空，但把握度必须为 low",
    )
    evidence: list[EvidenceItem] = Field(default_factory=list, description="当前判断依据（不要求穷尽）")
    missing_info: list[str] = Field(default_factory=list, description="还缺哪些关键信息（待排查清单）")
    next_steps: list[str] = Field(
        min_length=1,
        description="给患者的行动建议（观察/近日就医/尽快就医/立即急诊），至少 1 条",
    )
    confidence: Likelihood = Field(description="整体把握度；high 不等于确诊")


class CaseCard(BaseModel):
    """病历卡：每轮输出「更新后的完整快照」。"""

    age_group: str = Field(default="未提供", description="年龄区间（如 30-39）；不采集精确身份信息，未提供填「未提供」")
    sex: str = Field(default="未提供", description="男 / 女 / 其他 / 未提供")
    chief_complaint: str = Field(default="", description="主诉（一句话）")
    symptoms: list[str] = Field(default_factory=list, description="症状条目（部位、性质、程度、持续时间等已知属性）")
    timeline: list[str] = Field(default_factory=list, description="关键事件时间线（起病、演变、就医/用药），按时间排序")
    context: list[str] = Field(default_factory=list, description="既往史、过敏、长期用药、家族史、生活习惯等")
    red_flags_found: list[str] = Field(default_factory=list, description="已识别的危险信号")
    red_flags_excluded: list[str] = Field(default_factory=list, description="已确认排除的危险信号（需有依据）")
    hypotheses: list[Hypothesis] = Field(default_factory=list, description="与 assessment.hypotheses 同源的假设集")
    open_questions: list[str] = Field(default_factory=list, description="当前最值得问的问题候选")


class AgentTurnOutput(BaseModel):
    """本轮智能体输出：给患者的回复 + 结构化评估 + 病历卡快照 + 风险等级。"""

    reply: str = Field(description="给患者的最终回复（中文、短句、少术语；包含明确的行动建议；不得确诊、不得推荐药物与剂量）")
    assessment: Assessment
    case_card: CaseCard
    risk_level: RiskLevel = Field(description="low=可先观察 / medium=近期就医 / high=尽快(24小时内)就医 / emergency=立即急诊或拨打120")


class AgentResult(BaseModel):
    """answer() 对外返回结构（《智能体接口规范.md》§1.2）。"""

    reply: str
    assessment: Assessment
    risk_level: RiskLevel
    used_tools: list[str] = Field(default_factory=list, description="本轮实际调用的工具名（按调用顺序去重）")
    handled_as: HandledAs


class Evidence(BaseModel):
    """检索服务返回的证据片段（A 侧契约，见《智能体接口规范.md》§2.2）。"""

    chunk_id: str = Field(description="全局唯一且稳定的片段 ID")
    text: str = Field(description="片段正文（患者可读）")
    source: str = Field(default="", description="出处名称")
    section: str = Field(default="", description="章节/小节")
    score: float = Field(default=0.0, description="相关度分数（同一查询内可比，越大越相关）")
