# -*- coding: utf-8 -*-
"""OpenAI tools ↔ openPangu 提示词协议的桥接层（纯逻辑，不依赖 torch / NPU）。

openPangu-R-7B-2512 没有原生 function calling（chat template 只有
系统/助手/工具/方法/用户五种角色，没有 tools 参数），而 B 线智能体完全依赖
`tool_calls`（首轮就是 `tool_choice="required"`）。所以服务端要做两件事：

1. **工具轮**：把 OpenAI 的 tools 渲染进提示词，要求模型按两行协议输出
   `TOOL: <名字>` / `ARGS: <JSON>`，再解析回 OpenAI 的 tool_calls。
   实测失败模式是「少一个右花括号」「尾部多一个 }」，所以解析要容错
   （括号补全 + raw_decode 取第一个完整 JSON）。
2. **收口轮**：把 `AgentTurnOutput` 这种大嵌套结构单独用一次调用产出。
   实测让模型在工具轮里顺带吐大 JSON 几乎不可能成功，拆开之后一次就能成。
   收口是「格式转换」不是推理，因此这一轮固定用快思考。

本模块只做字符串/JSON 处理，可脱离 NPU 单元测试（见 tests/test_pangu_bridge.py）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

STRUCTURED_TOOL_NAME = "AgentTurnOutput"

TOOL_PROTOCOL = """【工具调用协议】（必须严格遵守，与普通对话无关）

可用工具：
{catalog}

输出规则：
1. 需要调用工具时，只输出下面两行，不要写解释、不要用代码块、不要加数组：
TOOL: <工具名>
ARGS: <一行 JSON 参数>
   例如：
TOOL: check_red_flags
ARGS: {"symptoms": ["胸痛", "大汗"]}
2. 用户**这一轮新提到症状**时，必须先调用 check_red_flags 核对危险信号，收到结果后再回答；
   涉及时间点、数字、资料依据时同理，先调用对应工具（timeline_calc / calculate_bmi /
   convert_units / retrieve_evidence），确实没有必要用工具时才直接回答。
3. 收到「工具：…返回：…」的消息后继续判断：还需要工具就再按两行输出一次；
   觉得信息已经够了，就直接用中文回答患者（不用输出工具调用），系统会替你整理结构化结果。
4. 一次只能输出一个调用，工具名必须是上面列出的名字。
5. 不要输出与上面格式无关的 JSON，也不要把多个调用拼在一起。
"""

FORCE_FINALIZE_NOTE = """
【本轮强制要求】你刚才没有按要求输出工具调用。请立刻做二选一：
- 还需要向用户补充提问 → 直接用中文问（系统会整理成结构化输出）；
- 信息已够 → 直接用中文给出本轮 reply（先共情，再给行动建议）。
不要再解释协议本身，也不要输出 JSON。
"""

REQUIRE_TOOL_NOTE = """
【本轮强制要求】你刚才没有输出工具调用。本轮必须先核对工具再回答：
- 用户这一轮描述了症状 → 调用 check_red_flags；
- 提到了时间点 / 体重身高 / 单位 / 需要资料 → 调用 timeline_calc / calculate_bmi /
  convert_units / retrieve_evidence。
请只输出 TOOL/ARGS 两行，不要输出中文回答。
"""

DEFAULT_FINALIZE_SYSTEM = """你是结构化输出转换器。用户会给你一段医患对话、工具结果和回复草稿，
请把它们整理成**一个 JSON 对象**，不要输出解释、不要用代码块、不要输出多个对象。

JSON 必须严格是这个结构（字段名与层级一模一样，只替换内容）：
{skeleton}

要求：
1. 只输出 JSON 本身，第一个字符必须是 {{；
2. reply 是给患者看的最终文本：先共情一句，再给行动建议；不得确诊、不得给药物名称与剂量；
3. assessment.hypotheses 按可能性降序，likelihood 只能取 low / medium / high；
4. assessment.evidence 的 source 只能取 dialogue / tool / kb，ref 要写清出处；
5. assessment.next_steps 至少一条；信息不足时给观察或就医建议；
6. assessment.confidence 只能取 low / medium / high；
7. case_card 是病历卡完整快照，没问到的一律写「未提供」；
8. risk_level 只能取 low / medium / high / emergency。
"""

OUTPUT_SKELETON: dict[str, Any] = {
    "reply": "给患者的回复文本",
    "assessment": {
        "hypotheses": [
            {
                "name": "假设名",
                "likelihood": "medium",
                "supporting": ["支持点"],
                "against": [],
                "missing": ["还缺的信息"],
            }
        ],
        "evidence": [{"source": "dialogue", "ref": "第1轮主诉", "detail": "患者自述……"}],
        "missing_info": ["还缺的信息"],
        "next_steps": ["给患者的行动建议"],
        "confidence": "low",
    },
    "case_card": {
        "age_group": "未提供",
        "sex": "未提供",
        "chief_complaint": "一句话主诉",
        "symptoms": ["症状条目"],
        "timeline": ["关键时间点"],
        "context": "未提供",
        "red_flags_found": [],
        "red_flags_excluded": [],
        "hypotheses": ["假设名"],
        "open_questions": ["下一轮要问的问题"],
    },
    "risk_level": "low",
}

_LIKELIHOOD_MAP = {
    "low": "low", "medium": "medium", "high": "high",
    "低": "low", "中": "medium", "中等": "medium", "高": "high",
}
_RISK_MAP = {
    "low": "low", "medium": "medium", "high": "high", "emergency": "emergency",
    "低": "low", "中": "medium", "中等": "medium", "高": "high",
    "紧急": "emergency", "急诊": "emergency", "危急": "emergency",
}
_SOURCE_MAP = {
    "dialogue": "dialogue", "tool": "tool", "kb": "kb",
    "对话": "dialogue", "问诊": "dialogue", "工具": "tool",
    "知识库": "kb", "检索": "kb", "资料": "kb",
}


@dataclass
class ParsedToolCall:
    """从模型输出里解析出来的工具调用。"""

    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class BridgeOutcome:
    """一次请求的桥接结果。"""

    kind: str  # "tool" | "final" | "text"
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# JSON 容错
# --------------------------------------------------------------------------
def repair_json(text: str) -> str:
    """补全被截断的右括号/引号（只做闭合，不改动已有内容）。"""
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    repaired = text + ('"' if in_string else "")
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _drop_dangling_tail(text: str) -> str | None:
    """截断在「键名刚写完、值还没写」时，去掉这半截键值对再闭合。"""
    trimmed = re.sub(r',?\s*"[^"]*"\s*:?\s*$', "", text)
    trimmed = trimmed.rstrip().rstrip(",")
    return trimmed if trimmed != text.strip() and trimmed else None


def load_json_tolerant(text: str) -> Any:
    """按「原样 → 补全括号 → 去掉末尾半截键值对再补全」依次尝试。

    尾部多余的 } ] 或文字由 raw_decode 忽略（实测模型常多写一个 }）。
    """
    decoder = json.JSONDecoder()
    body = text.strip()
    candidates = [body, repair_json(text).strip()]
    # 模型会照抄示例里的写法，把外层写成 {{...}}；这里顺手收敛一层
    if body.startswith("{{") and body.endswith("}}"):
        candidates.append(repair_json(body[1:-1]).strip())
    dropped = _drop_dangling_tail(body)
    if dropped:
        candidates.append(repair_json(dropped).strip())
    for candidate in candidates:
        if not candidate:
            continue
        try:
            value, _ = decoder.raw_decode(candidate)
            return value
        except json.JSONDecodeError:
            continue
    return None


def _iter_json_objects(text: str) -> Iterable[str]:
    """扫出所有平衡的 {…} 片段；未闭合的尾巴按补全后给出。"""
    cleaned = re.sub(r"```(?:json)?", "", text or "")
    for match in re.finditer(r"\{", cleaned):
        start = match.start()
        depth, in_string, escape = 0, False, False
        for index in range(start, len(cleaned)):
            ch = cleaned[index]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield cleaned[start : index + 1]
                    break
        else:
            yield repair_json(cleaned[start:])


# --------------------------------------------------------------------------
# 解析模型输出
# --------------------------------------------------------------------------
_TWO_LINE_RE = re.compile(
    r"TOOL\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*)\s*[\r\n]+?\s*ARGS\s*[:：]\s*(\{.*)",
    re.S,
)


def parse_tool_call(text: str, allowed: Sequence[str] | None = None) -> ParsedToolCall | None:
    """从模型输出里解析工具调用。

    支持协议规定的 `TOOL: x` / `ARGS: {...}`，以及模型偶尔直接吐的
    `{"tool_call": {"name": ..., "arguments": {...}}}`。
    """
    cleaned = re.sub(r"```(?:json)?", "", text or "")
    allowed_set = set(allowed) if allowed else None

    match = _TWO_LINE_RE.search(cleaned)
    if match:
        name, raw_args = match.group(1), match.group(2).strip()
        if allowed_set is None or name in allowed_set:
            args = load_json_tolerant(raw_args)
            if isinstance(args, dict):
                return ParsedToolCall(name=name, arguments=args, raw=match.group(0))

    for chunk in _iter_json_objects(cleaned):
        payload = load_json_tolerant(chunk)
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("tool_call"), dict):
            payload = payload["tool_call"]
        name = payload.get("name")
        args = payload.get("arguments")
        if not isinstance(name, str) or (allowed_set is not None and name not in allowed_set):
            continue
        if isinstance(args, str):
            args = load_json_tolerant(args)
        if isinstance(args, dict):
            return ParsedToolCall(name=name, arguments=args, raw=chunk)
    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    """收口轮用：从模型输出里取第一个 JSON 对象。"""
    cleaned = (text or "").strip()
    start = cleaned.find("{")
    if start < 0:
        return None
    value = load_json_tolerant(cleaned[start:])
    return value if isinstance(value, dict) else None


# --------------------------------------------------------------------------
# 归一化：把模型的自由发挥收敛到 B 线 schema
# --------------------------------------------------------------------------
_EMPTY_MARKERS = {"", "未提供", "无", "未知", "none", "null", "n/a", "不适用"}


def _as_str_list(value: Any) -> list[str]:
    """转成字符串列表，并丢掉「未提供/无/未知」这类占位值。"""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [] if text.lower() in _EMPTY_MARKERS else [text]
    if isinstance(value, (list, tuple)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip() and str(item).strip().lower() not in _EMPTY_MARKERS
        ]
    text = str(value).strip()
    return [] if text.lower() in _EMPTY_MARKERS else [text]


def _normalize_hypothesis(item: Any) -> dict[str, Any] | None:
    """把一条假设收敛成 B 线 Hypothesis 的形状；名字都没有的当垃圾丢掉。"""
    if isinstance(item, str):
        name = item.strip()
        return (
            {"name": name, "likelihood": "low", "supporting": [], "against": [], "missing": []}
            if name and name.lower() not in _EMPTY_MARKERS
            else None
        )
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "likelihood": _enum(item.get("likelihood"), _LIKELIHOOD_MAP, "low"),
        "supporting": _as_str_list(item.get("supporting")),
        "against": _as_str_list(item.get("against")),
        "missing": _as_str_list(item.get("missing")),
    }


def _enum(value: Any, mapping: dict[str, str], default: str) -> str:
    raw = str(value or "").strip()
    if raw.lower() in mapping:
        return mapping[raw.lower()]
    return mapping.get(raw, default)


def normalize_agent_output(payload: dict[str, Any], *, draft_reply: str = "") -> dict[str, Any]:
    """把模型产出的 JSON 收敛成 B 线 AgentTurnOutput 能通过校验的形状。

    只做「补默认值 / 映射枚举 / 丢垃圾字段」，不替模型编造医学内容：
    reply 缺失时用草稿或保守话术，next_steps 为空时补一条就医建议。
    """
    source = payload if isinstance(payload, dict) else {}
    assessment_raw = source.get("assessment") if isinstance(source.get("assessment"), dict) else {}
    card_raw = source.get("case_card") if isinstance(source.get("case_card"), dict) else {}

    hypotheses: list[dict[str, Any]] = []
    raw_hypotheses = assessment_raw.get("hypotheses")
    if isinstance(raw_hypotheses, list):
        for item in raw_hypotheses:
            normalized = _normalize_hypothesis(item)
            if normalized:
                hypotheses.append(normalized)

    evidence: list[dict[str, str]] = []
    raw_evidence = assessment_raw.get("evidence")
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            detail = str(item.get("detail") or "").strip()
            if not detail:
                continue
            evidence.append(
                {
                    "source": _enum(item.get("source"), _SOURCE_MAP, "dialogue"),
                    "ref": str(item.get("ref") or "").strip() or "本轮对话",
                    "detail": detail,
                }
            )

    next_steps = _as_str_list(assessment_raw.get("next_steps"))
    if not next_steps:
        next_steps = ["如症状加重或出现新的不适，请及时就医"]

    confidence = _enum(
        assessment_raw.get("confidence"),
        _LIKELIHOOD_MAP,
        "low" if not hypotheses else "medium",
    )

    reply = str(source.get("reply") or "").strip() or draft_reply.strip()
    if not reply:
        reply = (
            "抱歉，本轮我没能整理出可靠的分析。请您再描述一次最主要的症状和持续时间；"
            "如果情况较急，请直接就医。"
        )

    by_name = {item["name"]: item for item in hypotheses}
    raw_card_hypotheses = card_raw.get("hypotheses")
    if isinstance(raw_card_hypotheses, list) and raw_card_hypotheses:
        candidates: list[Any] = list(raw_card_hypotheses)
    elif isinstance(raw_card_hypotheses, str) and raw_card_hypotheses.strip().lower() not in _EMPTY_MARKERS:
        candidates = [raw_card_hypotheses]
    else:
        candidates = list(hypotheses)
    card_hypotheses: list[dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, dict):
            normalized = _normalize_hypothesis(item)
        else:
            normalized = by_name.get(str(item).strip()) or _normalize_hypothesis(item)
        if normalized:
            card_hypotheses.append(normalized)

    card = {
        "age_group": str(card_raw.get("age_group") or "未提供").strip(),
        "sex": str(card_raw.get("sex") or "未提供").strip(),
        "chief_complaint": str(card_raw.get("chief_complaint") or "").strip(),
        "symptoms": _as_str_list(card_raw.get("symptoms")),
        "timeline": _as_str_list(card_raw.get("timeline")),
        "context": _as_str_list(card_raw.get("context")),
        "red_flags_found": _as_str_list(card_raw.get("red_flags_found")),
        "red_flags_excluded": _as_str_list(card_raw.get("red_flags_excluded")),
        "hypotheses": card_hypotheses,
        "open_questions": _as_str_list(card_raw.get("open_questions")),
    }

    return {
        "reply": reply,
        "assessment": {
            "hypotheses": hypotheses,
            "evidence": evidence,
            "missing_info": _as_str_list(assessment_raw.get("missing_info")),
            "next_steps": next_steps,
            "confidence": confidence,
        },
        "case_card": card,
        "risk_level": _enum(source.get("risk_level"), _RISK_MAP, "medium"),
    }


# --------------------------------------------------------------------------
# 提示词组装
# --------------------------------------------------------------------------
def _first_sentence(text: str) -> str:
    body = " ".join(str(text).split())
    for stop in ("。", ". ", "；"):
        if stop in body:
            body = body.split(stop)[0] + "。"
            break
    return body[:120]


def _type_name(spec: Any) -> str:
    if not isinstance(spec, dict):
        return "any"
    if "enum" in spec:
        return "|".join(str(item) for item in spec["enum"])
    type_name = spec.get("type")
    if type_name == "array":
        return f"{_type_name(spec.get('items') or {})}[]"
    if type_name == "object":
        return "object"
    return str(type_name or "any")


def render_tool_catalog(tools: Sequence[dict] | None) -> str:
    """把 OpenAI tools 渲染成紧凑目录（完整 JSON Schema 太长，模型也读不动）。"""
    if not tools:
        return "（本轮没有可用工具）"
    lines: list[str] = []
    for tool in tools:
        function = (tool or {}).get("function") or {}
        name = function.get("name")
        if not name:
            continue
        description = _first_sentence(function.get("description") or "")
        params = function.get("parameters") or {}
        properties = params.get("properties") or {}
        required = set(params.get("required") or [])
        args = ", ".join(
            f"{key}: {_type_name(spec)}{'' if key in required else '（可选）'}"
            for key, spec in properties.items()
        )
        lines.append(f"- {name}({args})：{description}")
    return "\n".join(lines) if lines else "（本轮没有可用工具）"


def split_tools(tools: Sequence[dict] | None) -> tuple[list[dict], list[dict]]:
    """按名字把工具分成 (业务工具, 结构化输出工具)。"""
    business: list[dict] = []
    structured: list[dict] = []
    for tool in tools or []:
        name = ((tool or {}).get("function") or {}).get("name")
        if name == STRUCTURED_TOOL_NAME:
            structured.append(tool)
        else:
            business.append(tool)
    return business, structured


def build_tool_system_prompt(
    base_system: str,
    business_tools: Sequence[dict] | None,
    extra_notes: Sequence[str] = (),
) -> str:
    """工具轮的系统提示：B 线原提示词 + 工具协议 + 临时提示。"""
    protocol = TOOL_PROTOCOL.replace("{catalog}", render_tool_catalog(business_tools))
    parts = [base_system.strip(), protocol]
    parts.extend(note.strip() for note in extra_notes if note and note.strip())
    return "\n\n".join(part for part in parts if part)


def build_finalize_system() -> str:
    """收口轮的系统提示：不叠 B 线长提示词，只讲「怎么填这个结构」。"""
    skeleton = json.dumps(OUTPUT_SKELETON, ensure_ascii=False, indent=1)
    return DEFAULT_FINALIZE_SYSTEM.replace("{skeleton}", skeleton)


def collect_tool_names(messages: Sequence[dict]) -> dict[str, str]:
    """从历史里建立 tool_call_id → 工具名 的映射（OpenAI 的 tool 消息不带名字）。"""
    mapping: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call_id = str((call or {}).get("id") or "")
            name = ((call or {}).get("function") or {}).get("name")
            if call_id and name:
                mapping[call_id] = str(name)
    return mapping


def count_tool_rounds(messages: Sequence[dict]) -> int:
    """已经发生过多少次工具调用（用于限制轮数、控制延迟）。"""
    return sum(len(message.get("tool_calls") or []) for message in messages)


def _content_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):  # 兼容分块 content
        return "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") in (None, "text")
        )
    return str(content or "").strip()


def render_transcript(
    messages: Sequence[dict],
    *,
    tool_names_by_id: dict[str, str] | None = None,
    max_chars: int = 6000,
) -> str:
    """把 OpenAI 消息列表渲染成收口轮要读的「对话 + 工具结果」文本。"""
    names = tool_names_by_id or {}
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            lines.append(f"用户：{_content_of(message)}")
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            for call in calls:
                function = (call or {}).get("function") or {}
                lines.append(
                    f"[已调用工具] {function.get('name')} 参数 {function.get('arguments')}"
                )
            text = _content_of(message)
            if text:
                lines.append(f"助手：{text}")
        elif role == "tool":
            name = names.get(str(message.get("tool_call_id") or ""), "工具")
            lines.append(f"[工具结果] {name}：{_content_of(message)}")
    transcript = "\n".join(line for line in lines if line.strip())
    if len(transcript) > max_chars:
        transcript = "……（前文略）\n" + transcript[-max_chars:]
    return transcript


def build_finalize_user_prompt(transcript: str, draft_reply: str = "") -> str:
    """收口轮的 user 内容：对话记录 + 草稿 + 明确指令。"""
    parts = ["【本轮对话与工具结果】", transcript or "（无）"]
    if draft_reply.strip():
        parts += ["", "【助手上一轮写给患者的回复草稿】", draft_reply.strip()]
    parts += ["", "请按系统提示的结构输出 JSON。"]
    return "\n".join(parts)
