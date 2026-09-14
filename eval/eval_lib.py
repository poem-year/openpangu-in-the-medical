# -*- coding: utf-8 -*-
"""正式测评集（332 题 / 13 维度）的评测基础库：读题、拼提示词、解析、判分、统计。

设计要点
--------
1. **一套程序跑全库**：13 个维度字段统一，按 task_type 分派提示词与判分器。
2. **可复现**：贪心解码、固定提示词模板、固定批大小；配置指纹写进每条结果。
3. **分层**：layer 决定提示词风格，L0=裸模型直问，L1=加约束提示词；
   L2+ 需要 A 线检索 / B 线智能体接入，本库留出接口（`build_prompt` 里 raise 提示）。
"""

from __future__ import annotations

import difflib
import glob
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass

DATASET_DIR = "/data/openpangu/正式测评集"
PROMPT_VERSION = "v2"

# 各题型的输出预算（慢思考口径）。这些值按实测调过：
# 慢思考会先写完整推理再给结论，早先预算偏小导致 312 题里 180 题被截断、分数虚低。
MAX_NEW_TOKENS = {
    "mcq_single": 1536,
    "calculation": 3072,
    "evidence_qa": 1280,
    "open_diagnosis": 1536,
    "dialogue_diagnosis": 1536,
    "open_ddx": 1536,
    "open_ranked_ddx": 2048,
    "open_workup": 2048,
    "rubric_scored": 2560,
    "agentic_history_taking": 512,  # 需要模拟患者，L0 阶段跳过
    "retrieval": 512,               # 需要检索器，L0 阶段跳过
}

# L0 阶段跑不了、直接标注跳过的题型（原因写进结果，不静默丢弃）
SKIP_REASONS = {
    "agentic_history_taking": "需要先实现按隐藏信息作答的模拟患者（多轮交互），L0 阶段不跑",
    "retrieval": "金标准在 CmedqaRetrieval 语料里，本期只建业务知识库、未索引该语料，故跳过",
}

# 哪些层才有「知识库引用」：引用可追溯率是知识库模块的指标，基线（L0/L1）没有引用，
# 按约定统一记为 0，不用 D13 的材料引用去顶替。
# AGENT：整条链路交给 B 线智能体（它自己决定是否检索），所以同样统计引用可追溯率。
LAYERS_WITH_RAG = {"L2", "L3", "L5", "AGENT"}


def layer_has_rag(layer: str | None) -> bool:
    return (layer or "").upper() in LAYERS_WITH_RAG


# --------------------------------------------------------------------------
# 读题
# --------------------------------------------------------------------------
def load_items(
    dataset_dir: str = DATASET_DIR,
    dimensions: list[str] | None = None,
    levels: list[str] | None = None,
    task_types: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """按文件名顺序读入 13 个 jsonl；可按维度 / 层级 / 题型过滤。"""
    items: list[dict] = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if dimensions and item.get("dimension_code") not in dimensions:
                    continue
                if levels and not (set(item.get("level_hint") or []) & set(levels)):
                    continue
                if task_types and item.get("task_type") not in task_types:
                    continue
                items.append(item)
    if limit:
        items = items[:limit]
    return items


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------
def _format_options(options: dict | None) -> str:
    if not options:
        return ""
    return "\n".join(f"{k}. {v}" for k, v in options.items())


def format_rag_context(evidence: list[dict] | None) -> str:
    """把检索结果渲染成提示词里的参考资料块（含真实 chunk_id，供引用校验）。"""
    if not evidence:
        return (
            "【参考资料】\n（本轮未检索到相关资料。请如实说明没有查到资料，"
            "不要编造资料内容、出处或 chunk_id。）\n\n"
        )
    lines = ["【参考资料】（只能用下面这些资料作答，chunk_id 必须原样引用）"]
    for index, item in enumerate(evidence, start=1):
        source = item.get("source") or "未标注来源"
        section = item.get("section") or "未划分章节"
        lines.append(
            f"[{index}] 来源：{source}｜章节：{section}｜chunk_id：{item.get('chunk_id', '')}"
        )
        lines.append(str(item.get("text", "")).strip())
        lines.append("")
    lines.append(
        "引用要求：每条结论后面用 [chunk_id: <上面的 chunk_id>] 标注依据；"
        "资料里没有的内容不要写，也不要编造 chunk_id。"
    )
    return "\n".join(lines).strip() + "\n\n"


def build_prompt(item: dict, layer: str = "L0", rag_context: list[dict] | None = None) -> str:
    """把一道题拼成给模型的用户消息（提示词模板固定，版本记在结果里）。"""
    task = item["task_type"]
    question = item.get("question", "").strip()
    context = (item.get("context") or "").strip()
    dialogue = item.get("dialogue") or []

    body = ""
    if layer_has_rag(layer):
        body += format_rag_context(rag_context)
    if context:
        body += f"【病例/材料】\n{context}\n\n"
    if dialogue:
        lines = [f"{turn.get('role', '')}：{turn.get('text', '')}" for turn in dialogue]
        body += "【医患对话】\n" + "\n".join(lines) + "\n\n"

    if task == "mcq_single":
        return (
            f"{body}【问题】\n{question}\n\n【选项】\n{_format_options(item.get('options'))}\n\n"
            "请只选一个最合适的选项，最后一行按「答案：X」的格式给出选项字母。"
        )

    if task == "calculation":
        return (
            f"{body}【问题】\n{question}\n\n"
            "请完成计算，最后一行按「答案：<数值>」的格式给出数值结果（保留 4 位小数）。"
        )

    if task == "evidence_qa":
        return (
            f"{body}【问题】\n{question}\n\n"
            "请基于上面的材料回答：结论只能是 yes / no / maybe 之一，"
            "并指明结论依据来自哪一条摘要。最后一行格式：「答案：yes 依据：[摘要2]」。"
        )

    if task in ("open_diagnosis", "dialogue_diagnosis"):
        return (
            f"{body}【问题】\n{question}\n\n"
            "请给出最可能的诊断名称，并用一句话说明主要依据。"
            "最后一行按「诊断：<诊断名>」的格式给出结论。"
        )

    if task in ("open_ddx", "open_ranked_ddx"):
        return (
            f"{body}【问题】\n{question}\n\n"
            "请列出需要鉴别的疾病，按可能性从高到低排序，每行一个，"
            "格式为「序号. 疾病名」。"
        )

    if task == "open_workup":
        return (
            f"{body}【问题】\n{question}\n\n"
            "请列出为明确诊断需要优先进行的检查，每行一项，并说明各自目的。"
        )

    if task == "rubric_scored":
        # 原题本身就是向模型提出的用户问题，不加额外约束
        return f"{body}{question}"

    if task in SKIP_REASONS:
        raise NotImplementedError(SKIP_REASONS[task])

    return f"{body}【问题】\n{question}"


# --------------------------------------------------------------------------
# 文本解析
# --------------------------------------------------------------------------
_CN_PUNCT = "，。；：、（）()【】[]「」“”\"' \t\n\r·,.;:!?！？-—～~"


def normalize(text: str) -> str:
    text = (text or "").lower()
    for ch in _CN_PUNCT:
        text = text.replace(ch, "")
    return text


def parse_mcq_choice(text: str, options: dict | None) -> str | None:
    """从模型输出里抠出选项字母。"""
    letters = sorted((options or {}).keys()) or list("ABCDEFGHIJ")
    pattern = "".join(letters)
    m = re.findall(rf"答案\s*[:：]?\s*[（(]?\s*([{pattern}])\s*[)）]?", text, flags=re.I)
    if m:
        return m[-1].upper()
    tail = text[-300:]
    hits = re.findall(rf"(?:^|[^A-Za-z])([{pattern}])(?![A-Za-z])", tail)
    if hits:
        return hits[-1].upper()
    return None


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_number(text: str) -> float | None:
    """抠出数值答案：优先「答案：」后面的数，再看 ≈/=，最后取末行数字。"""
    m = re.findall(r"答案\s*[:：]?\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            pass
    for pattern in (r"[≈=]\s*(-?\d+(?:\.\d+)?)",):
        hits = re.findall(pattern, text)
        if hits:
            try:
                return float(hits[-1])
            except ValueError:
                pass
    lines = [l for l in (text or "").splitlines() if l.strip()]
    for line in reversed(lines):
        nums = _NUM_RE.findall(line)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                continue
    return None


def numbers_match(got: float | None, want: float | None, rel: float = 0.01, abs_tol: float = 0.01) -> bool:
    if got is None or want is None:
        return False
    if abs(got - want) <= abs_tol:
        return True
    if want == 0:
        return False
    return abs(got - want) / abs(want) <= rel


def bigram_dice(a: str, b: str) -> float:
    """字符二元组 Dice 相似度（中文没有空格，按字面二元组算）。"""
    a, b = normalize(a), normalize(b)
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a and a == b else 0.0
    ga = {a[i:i + 2] for i in range(len(a) - 1)}
    gb = {b[i:i + 2] for i in range(len(b) - 1)}
    inter = len(ga & gb)
    return 2 * inter / (len(ga) + len(gb))


def similarity(needle: str, haystack: str) -> float:
    """相似度：包含关系算满分，否则取二元组 Dice。"""
    n, h = normalize(needle), normalize(haystack)
    if not n or not h:
        return 0.0
    if n in h:
        return 1.0
    dice = bigram_dice(n, h)
    if dice >= 0.75:
        return dice
    # 长要点用最长公共子串兜底（避免长句被 Dice 稀释）
    matcher = difflib.SequenceMatcher(None, n, h, autojunk=False)
    match = matcher.find_longest_match(0, len(n), 0, len(h))
    return max(dice, match.size / len(n))


def fuzzy_contains(needle: str, haystack: str, threshold: float = 0.5) -> bool:
    return similarity(needle, haystack) >= threshold


def _core_phrase(text: str, max_len: int = 30) -> str:
    """把「①xx:yy」这类条目压成核心短语。"""
    t = re.sub(r"^[（(]?\d+[）)]?|[①-⑩]", "", text or "").strip()
    t = re.split(r"[:：，,。；;（(]", t)[0].strip()
    return t[:max_len]


def parse_inline_options(question: str) -> dict:
    """题干里内嵌的 A. xx / B. xx 选项（D04 的开放题会这么写）。"""
    opts = {}
    for m in re.finditer(r"(?:^|\s|　)([A-J])[.、．]\s*([^\n]+?)(?=(?:\s+[A-J][.、．])|\n|$)", question or ""):
        opts[m.group(1)] = m.group(2).strip()
    return opts if len(opts) >= 2 else {}


def extract_reference_letter(reference: str) -> str | None:
    m = re.search(r"答案\s*(?:是|为|[:：])?\s*([A-J])", reference or "")
    return m.group(1).upper() if m else None


def extract_diagnosis(item: dict) -> str | None:
    """从 key_points / reference_answer 里抽出期望的主诊断名。"""
    key_points = item.get("key_points") or []
    for kp in key_points[:2]:
        m = re.search(r"诊断\s*[:：]\s*([^\n。；;（(]+)", kp)
        if m:
            return _core_phrase(m.group(1), 25)
    ref = (item.get("reference_answer") or "").strip()
    if ref:
        first = re.split(r"[。\n；;]", ref)[0]
        first = re.sub(r"^(诊断|结论)\s*[:：]?\s*", "", first).strip()
        return _core_phrase(first, 25) or None
    return None


def _split_key_points(text: str) -> list[str]:
    text = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", "\n", text or "")
    parts = re.split(r"[\n；;]|(?<=[。;；])\s*", text)
    return [p.strip(" -·•\t") for p in parts if len(p.strip()) >= 2]


def extract_ddx_list(item: dict) -> list[str]:
    """鉴别诊断期望列表：从 key_points 拆条目，再压掉前缀说明。"""
    source = item.get("key_points") or [item.get("reference_answer") or ""]
    out: list[str] = []
    for kp in source:
        for part in _split_key_points(kp):
            core = _core_phrase(part, 20)
            if 2 <= len(core) <= 20:
                out.append(core)
            # 「如食管憩室」这类括注里的具体病名也算一条
            for extra in re.findall(r"如([一-龥A-Za-z]{2,12})", part):
                out.append(extra)
    # 去重保序
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# --------------------------------------------------------------------------
# 判分
# --------------------------------------------------------------------------
@dataclass
class Score:
    item_id: str
    dimension: str
    task_type: str
    score: float | None          # 0-1，None 表示需要人工/模型判分
    correct: bool | None
    metrics: dict
    detail: str = ""


def score_item(item: dict, output: str, thinking: str = "") -> Score:
    """按题型判分。output 为模型正文（不含思考过程）。"""
    task = item["task_type"]
    text = output or ""
    combined = (output or "") + "\n" + (thinking or "")
    metrics: dict = {}

    if task == "mcq_single":
        got = parse_mcq_choice(text, item.get("options"))
        want = (item.get("answer") or "").strip().upper()
        correct = got == want
        metrics["choice"] = got
        return Score(item["id"], item["dimension_code"], task, 1.0 if correct else 0.0,
                     correct, metrics, f"预测 {got} / 正确 {want}")

    if task == "calculation":
        got = extract_number(text) or extract_number(combined)
        want = extract_number(str(item.get("reference_answer") or ""))
        ok = numbers_match(got, want)
        metrics["value"] = got
        return Score(item["id"], item["dimension_code"], task, 1.0 if ok else 0.0,
                     ok, metrics, f"预测 {got} / 正确 {want}")

    if task == "evidence_qa":
        want = (item.get("answer") or "").strip().lower()
        m = re.search(r"答案\s*[:：]?\s*(yes|no|maybe|是|否|可能)", text, flags=re.I)
        got = m.group(1).lower() if m else None
        alias = {"是": "yes", "否": "no", "可能": "maybe"}
        got = alias.get(got, got)
        cites = re.findall(r"\[摘要\s*(\d+)\]", text)
        ctx = item.get("context") or ""
        n_abs = len(re.findall(r"\[摘要\s*\d+\]", ctx))
        cite_ok = bool(cites) and all(int(c) <= max(n_abs, 1) for c in cites)
        metrics.update(choice=got, citations=cites, citation_traceable=cite_ok)
        return Score(item["id"], item["dimension_code"], task, 1.0 if got == want else 0.0,
                     got == want, metrics, f"预测 {got} / 正确 {want}；引用 {cites or '无'}")

    if task in ("open_diagnosis", "dialogue_diagnosis"):
        want = extract_diagnosis(item)
        sim = similarity(want, combined) if want else 0.0
        hit = sim >= 0.5
        kps = item.get("key_points") or []
        cov = sum(1 for kp in kps if fuzzy_contains(kp, combined)) / len(kps) if kps else None
        metrics.update(diagnosis=want, diagnosis_similarity=round(sim, 3), key_point_coverage=cov)
        return Score(item["id"], item["dimension_code"], task, 1.0 if hit else 0.0, hit,
                     metrics, f"期望诊断 {want}；相似度 {sim:.2f}；要点覆盖 {cov}")

    if task in ("open_ddx", "open_ranked_ddx"):
        expects = extract_ddx_list(item)
        hits = [e for e in expects if fuzzy_contains(e, combined)]
        recall = len(hits) / len(expects) if expects else None
        mrr = 0.0
        if task == "open_ranked_ddx" and expects:
            ranked = [l for l in text.splitlines() if l.strip()]
            for rank, line in enumerate(ranked, start=1):
                if any(fuzzy_contains(e, line) for e in expects):
                    mrr = 1.0 / rank
                    break
        metrics.update(expected=expects, hit=hits, recall=recall,
                       mrr=(mrr if task == "open_ranked_ddx" else None))
        return Score(item["id"], item["dimension_code"], task, recall, None, metrics,
                     f"期望 {len(expects)} 项，命中 {len(hits)} 项")

    if task == "open_workup":
        # D04 的开放题多半在题干里内嵌了 A/B/C/D 选项，参考答案里写了「答案是X」，
        # 这种就按选项判分，比要点覆盖稳。其余按要点覆盖。
        inline = parse_inline_options(item.get("question", ""))
        ref_letter = extract_reference_letter(item.get("reference_answer", ""))
        if inline and ref_letter:
            got = parse_mcq_choice(text, inline)
            ok = got == ref_letter
            metrics.update(choice=got, reference_choice=ref_letter,
                           options=list(inline.keys()))
            return Score(item["id"], item["dimension_code"], task, 1.0 if ok else 0.0, ok,
                         metrics, f"预测 {got} / 参考 {ref_letter}")
        kps = item.get("key_points") or []
        sentences: list[str] = []
        for kp in kps:
            sentences.extend(s for s in re.split(r"[。；;\n]", kp) if len(s.strip()) >= 6)
        cov = (sum(1 for s in sentences if fuzzy_contains(s, combined, threshold=0.45)) / len(sentences)
               if sentences else None)
        metrics["key_point_coverage"] = cov
        return Score(item["id"], item["dimension_code"], task, cov, None, metrics,
                     f"要点覆盖 {cov}")

    if task == "rubric_scored":
        return Score(item["id"], item["dimension_code"], task, None, None, {},
                     "需要 rubric 判分（脚本判分阶段或人工复核）")

    return Score(item["id"], item["dimension_code"], task, None, None, {},
                 SKIP_REASONS.get(task, "未实现的题型") + "（待人工判分）")


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def config_fingerprint(**kwargs) -> str:
    payload = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
