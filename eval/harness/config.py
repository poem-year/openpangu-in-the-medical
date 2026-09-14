"""实验配置：一份 YAML 描述一次评测「怎么跑」，框架据此拼命令、算指纹、写 meta。

引擎只有两种（都在 `eval/` 下，框架不重写算法）：
- `direct`     ：`eval/run_eval.py`，评测端拼提示词直接喂模型（L0–L5）
- `agent-batch`：`eval/run_agent_batch.py`，走智能体状态机、按轮次批处理

字段（除 name/engine 外都有默认值）：
    name           实验名（进 run id，短横线英文）
    engine         direct | agent-batch
    layer          L0/L1/L2/L3/L5 或 AGENT（agent-batch 固定用 AGENT）
    thinking       slow | fast（agent-batch 忽略）
    budget_scale   输出预算倍数（1.0 = 基准；4.0 = 四倍）
    retrieval_k    RAG 每题材条数（agent-batch 的 RAG 信息量旋钮）
    batch_size     批大小
    per_dimension  每维度取前 N 题（冒烟；0/省略 = 全量）
    dimensions     只跑指定维度（省略 = 全部）
    limit          总题数上限（省略 = 不限）
    judge          是否跑 AI 判分（rubric + 开放题复核）
    judge_workers  判分并发
    notes          备注（写进 meta 与登记表）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "eval" / "configs"
VALID_ENGINES = ("direct", "agent-batch")


@dataclass
class ExperimentConfig:
    name: str
    engine: str = "direct"
    layer: str = "L0"
    thinking: str = "slow"
    budget_scale: float = 1.0
    retrieval_k: int = 3
    batch_size: int = 32
    per_dimension: int = 0
    dimensions: list[str] = field(default_factory=list)
    limit: int = 0
    judge: bool = True
    judge_workers: int = 6
    notes: str = ""

    def validated(self) -> "ExperimentConfig":
        if self.engine not in VALID_ENGINES:
            raise ValueError(f"engine 必须是 {VALID_ENGINES} 之一，收到 {self.engine!r}")
        if self.engine == "agent-batch":
            self.layer = "AGENT"
        if self.budget_scale <= 0:
            raise ValueError("budget_scale 必须 > 0")
        if self.retrieval_k < 1:
            raise ValueError("retrieval_k 必须 ≥ 1")
        if self.batch_size < 1:
            raise ValueError("batch_size 必须 ≥ 1")
        return self

    def fingerprint_payload(self) -> dict:
        """进 run id 指纹的字段：改了任意一个，就是一次新实验。"""
        return {
            "engine": self.engine, "layer": self.layer, "thinking": self.thinking,
            "budget_scale": self.budget_scale, "retrieval_k": self.retrieval_k,
            "batch_size": self.batch_size, "per_dimension": self.per_dimension,
            "dimensions": sorted(self.dimensions), "limit": self.limit,
            "judge": self.judge, "judge_workers": self.judge_workers,
        }

    def param_summary(self) -> str:
        bits = [f"budget×{self.budget_scale:g}", f"batch{self.batch_size}"]
        if self.engine == "agent-batch":
            bits.append(f"k={self.retrieval_k}")
        if self.per_dimension:
            bits.append(f"每维{self.per_dimension}题")
        if self.limit:
            bits.append(f"上限{self.limit}题")
        if self.dimensions:
            bits.append("维度" + "/".join(self.dimensions))
        bits.append("AI判分" if self.judge else "仅规则判分")
        return "、".join(bits)


def load_config(path: Path | str) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置必须是键值表：{path}")
    if not data.get("name"):
        raise ValueError(f"配置缺少 name：{path}")
    allowed = set(ExperimentConfig.__dataclass_fields__)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"配置里有不认识的字段 {unknown}（可用：{sorted(allowed)}）")
    return ExperimentConfig(**data).validated()


def available_configs() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yml")) + sorted(CONFIG_DIR.glob("*.yaml"))
