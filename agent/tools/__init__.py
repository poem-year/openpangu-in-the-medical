"""本地工具集（4 个本地工具 + 1 个检索工具）。"""

from agent.retrieval import retrieve_evidence
from agent.tools.bmi import calculate_bmi
from agent.tools.red_flags import check_red_flags
from agent.tools.timeline import timeline_calc
from agent.tools.units import convert_units

ALL_TOOLS = [calculate_bmi, convert_units, check_red_flags, timeline_calc, retrieve_evidence]

__all__ = [
    "ALL_TOOLS",
    "calculate_bmi",
    "check_red_flags",
    "convert_units",
    "retrieve_evidence",
    "timeline_calc",
]
