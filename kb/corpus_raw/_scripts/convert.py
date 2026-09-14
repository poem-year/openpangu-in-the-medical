"""语料预处理：corpus_raw → kb/corpus（Markdown + sources.yaml）。

为什么需要这一层：`kb.build` 只收 `.md/.txt/.pdf`，且出处只认 `sources.yaml`。
原始层是 4.5 GB 的 html/xml/jsonl/docx/zip，必须先转成「带标题层级的纯文本」，
否则切块只能按 500 字硬切，section 全空、引用指不到具体条款。

本脚本做四件事：
  1. 格式转换：html / xml / jsonl / docx / pdf → Markdown（带标题层级）；
  2. 内容清洗：按关键词与规则剔除「检索混进来的非医学文件」（招聘表、专业目录、
     非医学政策），剔掉的每一份都写进 conversion_report.json，不静默跳过；
  3. 结构对齐：把「一、（一）、第X条」转成标题层级，section 才有意义；
  4. 登记：生成 `kb/corpus/sources.yaml`（含 license / language / tier）。

用法：
    .venv/bin/python kb/corpus_raw/_scripts/convert.py             # 转换 + 写登记表
    .venv/bin/python kb/corpus_raw/_scripts/convert.py --dry-run   # 只看统计不落盘
    .venv/bin/python kb/corpus_raw/_scripts/convert.py --count-chunks
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

RAW = Path(__file__).resolve().parents[1]
REPO = RAW.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_OUT = REPO / "kb" / "corpus"


@dataclass
class Doc:
    """一份待入库的语料（转换后的 Markdown）。"""

    rel_path: str
    doc_id: str
    name: str
    text: str
    category: str
    source_type: str
    version: str = ""
    url: str = ""
    note: str = ""
    license: str = ""
    language: str = "zh"
    tier: str = "3"
    origin: str = ""


@dataclass
class Report:
    kept: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def keep(self, doc: "Doc", chars: int) -> None:
        self.kept.append(
            {
                "file": doc.rel_path,
                "doc_id": doc.doc_id,
                "name": doc.name,
                "tier": doc.tier,
                "license": doc.license,
                "language": doc.language,
                "chars": chars,
                "origin": doc.origin,
            }
        )

    def skip(self, origin: str, reason: str, detail: str = "") -> None:
        self.skipped.append({"origin": origin, "reason": reason, "detail": detail[:200]})


_WS_RE = re.compile(r"[ \t\u3000\u00a0]+")
_BLANK_RE = re.compile(r"\n{3,}")


def norm_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 百科与 docx 正文里混着 &gt; / &amp; 这类实体，留着会脏了检索到的证据文本
    text = html_mod.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_RE.sub("\n\n", text).strip()


def strip_tags(fragment: str) -> str:
    fragment = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", fragment)
    fragment = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|table|section|article)>", "\n", fragment)
    fragment = re.sub(r"(?i)<li[^>]*>", "\n· ", fragment)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return norm_text(html_mod.unescape(fragment))


def clean_title(raw: str) -> str:
    title = norm_text(strip_tags(raw))
    pattern = re.compile(
        r"\s*[_|｜]\s*(中国政府网|国务院(?:部门)?文件|.*?卫生健康委员会|.*?疾病预防控制中心)\s*$"
    )
    for _ in range(3):  # 标题常见「正文名_国务院文件_中国政府网」多级后缀
        new = pattern.sub("", title)
        if new == title:
            break
        title = new
    return title.strip(" _|｜")


def slug(text: str, maxlen: int = 40) -> str:
    """文件名用：保留中英文数字，其余压成下划线。"""
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text).strip("_")
    return (text[:maxlen] or "doc").strip("_")


def md_hash(*parts: str) -> str:
    return hashlib.md5("\u0000".join(parts).encode("utf-8")).hexdigest()[:8]


_CONTAINER_PATTERNS = (
    r'<div[^>]+id="UCAP-CONTENT"[^>]*>',
    r'<div[^>]+class="[^"]*Article_content[^"]*"[^>]*>',
    r'<div[^>]+class="[^"]*trout-region-content[^"]*"[^>]*>',
    r'<div[^>]+class="[^"]*TRS_Editor[^"]*"[^>]*>',
    r'<div[^>]+id="zoom"[^>]*>',
    r"<article[^>]*>",
)


def _extract_balanced(html: str, start_idx: int) -> str:
    """从 start_idx 处的开标签起，按同名标签配对取正文片段（够用即可）。"""
    tag_match = re.match(r"<(\w+)", html[start_idx:])
    if not tag_match:
        return html[start_idx:]
    tag = tag_match.group(1).lower()
    depth = 0
    pos = start_idx
    pattern = re.compile(rf"(?i)</?{tag}\b")
    while True:
        match = pattern.search(html, pos)
        if not match:
            return html[start_idx:]
        if html[match.start() : match.start() + 2] == "</":
            depth -= 1
            if depth == 0:
                return html[start_idx : match.end()]
        else:
            depth += 1
        pos = match.end()


def html_body(html: str, *, require_container: bool = False) -> str:
    """取网页正文：先按已知容器定位；找不到容器时返回整页（除非要求必须有容器）。"""
    for pattern in _CONTAINER_PATTERNS:
        match = re.search(pattern, html, re.I)
        if match:
            body = strip_tags(_extract_balanced(html, match.start()))
            if len(body) >= 200:
                return body
    return "" if require_container else strip_tags(html)


def html_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    return clean_title(match.group(1)) if match else ""


_CH_NUM = "一二三四五六七八九十百零"
_CLAUSE_RE = re.compile(
    rf"^(?:(第[{_CH_NUM}]+[章节])|([{_CH_NUM}]+、)|(（[{_CH_NUM}]+）)|(第[{_CH_NUM}]+条))(.{{0,40}})$"
)


def _heading_level(marker: str) -> int:
    if marker.startswith("第") and marker.endswith("章"):
        return 2
    if marker.startswith("第") and marker.endswith("节"):
        return 3
    if marker.endswith("、"):
        return 2
    if marker.endswith("条"):
        return 3
    return 4


def structure_markdown(title: str, body: str, *, allow_clause_headings: bool = True) -> str:
    """把正文按条款标记切成 `## / ###` 标题，供 kb 的标题切块使用。

    单独成行的「一、」「（一）」才当标题（行内含正文的按段落保留）；
    「第X条」条目过多时不再升级为标题，避免把文档切成一堆百字碎块。
    """
    lines = body.split("\n")
    n_clause = sum(1 for line in lines if re.match(rf"^第[{_CH_NUM}]+条", line.strip()))
    use_clause = allow_clause_headings and n_clause <= 60

    blocks: list[list] = [[2, "正文", []]]
    for line in lines:
        stripped = line.strip()
        match = _CLAUSE_RE.match(stripped) if stripped else None
        if match:
            marker = next(group for group in match.groups()[:4] if group)
            rest = match.group(5).strip()
            if not rest or len(rest) <= 30:
                if marker.endswith("条") and not use_clause:
                    blocks[-1][2].append(line)
                    continue
                blocks.append([_heading_level(marker), (marker + " " + rest).strip(), []])
                continue
        blocks[-1][2].append(line)

    out: list[str] = [f"# {title}", ""]
    pending_head: str | None = None
    pending_body: list[str] = []

    def flush() -> None:
        if pending_body:
            out.append(f"## {pending_head}" if pending_head else "## 正文")
            out.append("")
            out.extend(pending_body)
            out.append("")
        pending_body.clear()

    for _level, head, lines_ in blocks:
        body_text = norm_text("\n".join(lines_))
        if not body_text:
            continue
        if pending_head is None:
            pending_head = head
        elif len("\n".join(pending_body)) >= 200:
            flush()
            pending_head = head
        else:
            # 上一节太短：合并，小标题文本保留在正文里（检索与阅读都还需要它）
            pending_body.append(f"**{head}** {body_text}")
            continue
        pending_body.append(body_text)
    flush()
    return norm_text("\n".join(out))


_STRONG_MED = (
    "医", "药", "病", "诊", "疗", "健康", "卫生", "临床", "疫苗", "肿瘤", "症",
    "护理", "康复", "疾控", "中医", "营养", "血压", "血糖", "传染", "职业病",
    "放射诊疗", "精神卫生", "母婴", "儿童", "老年", "急救", "输血", "手术",
)
_NOISE_TITLE = re.compile(
    r"招聘|应聘|申请表|登记表|个人信息|专业对照|自学考试|继续教育|教学点|专业设置|"
    r"课程|名单|一览表|公示|评审结果|结果通知|申报|立项|税费|供水|阅读|农产品|"
    r"林业|草原|汽车|家政|旅行服务|境外人员|污水|臭氧|捕捞|外资|商业|消费|"
    r"农业保险|多式联运|住房|公积金|信用修复|养老服务|托育|儿童友好|人工智能|"
    r"人事|会计|统计|信访|档案|"
    # 省市级文件里混进来的行政/办事类附件（与临床无关）
    r"操作手册|使用手册|登记地点|咨询电话|参考版式|协议书|合同|考核操作流程|"
    r"评审标准|整改表|自查|培训基地|学员|招生|计划拟立项|平台|"
    r"招收|招募|聘用|博士后|简章|启事|"
    # 医学相关但属产业/宣传类，不是诊断依据
    r"师承|传承教育|确有专长|数智化|工业|零售|标准备案|食品企业|"
    r"科普活动|主题宣传|世界卒中日|启动仪式|海洋药物"
)
# 标题必须命中的临床关键词：拦掉「殡葬管理条例」「妇女儿童发展纲要」这类
# 正文偶发提到「医疗卫生」但主题与临床无关的文件
_TITLE_MED = (
    "医", "药", "病", "诊", "疗", "健康", "卫生", "临床", "疫苗", "肿瘤", "症",
    "护理", "康复", "疾控", "中医", "营养", "血压", "血糖", "传染", "职业病",
    "放射诊疗", "精神卫生", "急救", "输血", "手术", "分娩", "镇痛", "产前",
    "残疾预防", "免疫规划", "中毒", "伤害",
)
# 标题必须命中的临床主题词：把「体制改革」「医院建设」「设备采购」这类管理类
# 政策挡在库外——它们是政策依据，不是诊断依据
_TITLE_TOPIC = (
    "诊疗", "诊治", "防治", "诊断", "治疗", "用药", "药物", "药品", "疫苗",
    "免疫规划", "筛查", "护理", "急救", "急症", "转诊", "首诊", "分级诊疗",
    "基本病种", "职业病", "传染病", "慢性病", "心脑血管", "艾滋病", "结核",
    "肿瘤", "癌", "糖尿", "高血压", "家庭病床", "医养", "儿科", "精神卫生",
    "老年医学", "分娩", "镇痛", "产科", "新生儿", "罕见病", "处方", "抗菌",
    "输血", "感染", "中毒", "临床", "就医", "医院感染", "疾病预防控制",
    "公共卫生", "健康管理", "营养", "残疾预防", "医学", "中医", "中医药", "中药",
)


def medical_score(title: str, body: str) -> int:
    score = sum(3 for kw in _STRONG_MED if kw in title)
    head = body[:1500]
    score += sum(1 for kw in _STRONG_MED if kw in head)
    for kw in ("诊断", "治疗", "症状", "用药", "禁忌", "适应症", "不良反应", "鉴别"):
        if kw in title:
            score += 2
        elif kw in head:
            score += 1
    return score


def is_medical(
    title: str,
    body: str,
    *,
    threshold: int = 4,
    check_body_head: bool = False,
) -> tuple[bool, str]:
    if not title:
        return False, "标题为空"
    noise = _NOISE_TITLE.search(title)
    if noise:
        return False, f"标题命中非医学模式：{noise.group(0)}"
    if check_body_head:
        head_noise = _NOISE_TITLE.search(body[:300])
        if head_noise:
            return False, f"正文开头命中非医学模式：{head_noise.group(0)}"
    if not any(kw in title for kw in _TITLE_MED):
        return False, "标题无临床关键词（主题不是临床/卫生）"
    if not any(kw in title for kw in _TITLE_TOPIC):
        return False, "标题无临床主题词（政策/产业/管理类，不宜作为诊断依据）"
    if len(body) < 400:
        return False, f"正文过短（{len(body)} 字）"
    score = medical_score(title, body)
    if score < threshold:
        return False, f"医学相关度不足（score={score}）"
    return True, f"score={score}"


def docx_text(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    # 先按表格行切开（表格是政府附件的主要形态），再按段落切开
    rows = xml.replace("</w:tr>", "\n@@ROW@@")
    lines: list[str] = []
    for chunk in rows.split("@@ROW@@"):
        cells = chunk.split("</w:tc>")
        if len(cells) > 1:
            values = [_runs_text(cell) for cell in cells]
            line = " | ".join(value for value in values if value)
            if line.strip(" |"):
                lines.append(line)
            continue
        for paragraph in chunk.split("</w:p>"):
            value = _runs_text(paragraph)
            if value:
                lines.append(value)
    text = norm_text("\n".join(lines))
    title = text.split("\n", 1)[0][:80] if text else ""
    return text, title


_W_RUN_RE = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)


def _runs_text(fragment: str) -> str:
    runs = [html_mod.unescape(item) for item in _W_RUN_RE.findall(fragment)]
    return norm_text("".join(runs))


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    raw = "\n".join((page.extract_text() or "") for page in reader.pages)
    return fix_pdf_layout(raw)


_PAGE_NO_RE = re.compile(r"^[—\-–\s]*\d{1,3}[—\-–\s]*$")
_SENT_END = "。；！？：）」』”\"'"


def fix_pdf_layout(text: str) -> str:
    """PDF 抽取会有页码行与句中硬换行，按标点回接，否则切块会碎成半句话。"""
    kept: list[str] = []
    headings: list[str] = []
    for line in text.replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped or _PAGE_NO_RE.match(stripped):
            continue
        if _CLAUSE_RE.match(stripped) or stripped.startswith(("附件", "第", "一、", "二、")):
            if len(stripped) <= 40:
                headings.append(stripped)
                kept.append(stripped)
                continue
        if kept and not kept[-1].endswith(tuple(_SENT_END)):
            kept[-1] = kept[-1] + stripped
        else:
            kept.append(stripped)
    return norm_text("\n".join(kept))


def derive_title(text: str, fallback: str) -> str:
    """附件类 PDF/docx 没有 <title>：从正文头部挑一行最像标题的。"""
    lines = [line.strip() for line in text.split("\n") if line.strip()][:30]
    for line in lines:
        match = re.match(r"^附件\s*\d*\s*[：:、.]?\s*(.+)$", line)
        if match and len(match.group(1).strip()) >= 5:
            return match.group(1).strip()[:80]
    for line in lines:
        if len(line) < 6 or _PAGE_NO_RE.match(line):
            continue
        if sum(1 for ch in line if "\u4e00" <= ch <= "\u9fff") < 5:
            continue
        if line.endswith(("：", ":", "。")) or line.startswith(("目录", "目次")):
            continue
        return line[:80]
    return fallback


def read_any(path: Path) -> tuple[str, str]:
    """读一份原始文件，返回 (正文, 标题)。"""
    suffix = path.suffix.lower()
    if suffix == ".html":
        raw = path.read_text(encoding="utf-8", errors="replace")
        # 政府站点的列表页/导航页抽不到正文容器，这类页面直接不入库
        return html_body(raw, require_container=True), html_title(raw)
    if suffix == ".docx":
        text, title = docx_text(path)
        if len(re.sub(r"[^\w\u4e00-\u9fff]", "", title)) < 5:
            title = derive_title(text, path.stem)
        return text, title
    if suffix == ".pdf":
        text = pdf_text(path)
        return text, derive_title(text, path.stem)
    raise ValueError(f"不支持的格式：{suffix}")


def build_md_doc(
    *,
    rel_path: str,
    doc_id: str,
    title: str,
    body: str,
    category: str,
    source_type: str,
    version: str,
    url: str,
    note: str,
    license_text: str,
    language: str,
    tier: str,
    origin: str,
) -> Doc:
    return Doc(
        rel_path=rel_path,
        doc_id=doc_id,
        name=title,
        text=structure_markdown(title, body),
        category=category,
        source_type=source_type,
        version=version,
        url=url,
        note=note,
        license=license_text,
        language=language,
        tier=tier,
        origin=origin,
    )


def adapter_gov_cn(report: Report) -> list[Doc]:
    """中国政府网文件库：169 篇 html + 政府附件（docx/pdf）。"""
    docs: list[Doc] = []
    root = RAW / "guidelines_cn" / "gov_cn"
    for path in sorted(root.iterdir()):
        origin = path.relative_to(RAW).as_posix()
        if path.suffix.lower() not in (".html", ".docx", ".pdf"):
            report.skip(origin, "格式暂不转换（.doc/.wps 需额外工具且临床价值低）")
            continue
        name_noise = _NOISE_TITLE.search(path.stem)
        if name_noise:
            report.skip(origin, f"文件名命中非医学模式：{name_noise.group(0)}")
            continue
        try:
            body, title = read_any(path)
        except Exception as exc:  # noqa: BLE001
            report.skip(origin, f"读取失败：{type(exc).__name__}: {exc}")
            continue
        if not body:
            reason = (
                "未定位到正文容器（列表/导航页，非文章正文）"
                if path.suffix.lower() == ".html"
                else "解析后无正文（疑似扫描件/空文件）"
            )
            report.skip(origin, reason)
            continue
        if not title:
            title = body.split("\n", 1)[0][:60]
        ok, why = is_medical(title, body, check_body_head=True)
        if not ok:
            report.skip(origin, why)
            continue
        doc = build_md_doc(
            rel_path=f"guidelines_cn/gov_cn/{slug(title)}_{md_hash(origin)}.md",
            doc_id=f"gov-{md_hash(origin)}",
            title=title,
            body=body,
            category="国家政策文件",
            source_type="政府文件",
            version="2026-09 采集",
            url="",
            note="来源：中国政府网文件库，正文由 html/docx/pdf 转换",
            license_text="政府文件（可公开，注明来源与文号）",
            language="zh",
            tier="1",
            origin=origin,
        )
        docs.append(doc)
        report.keep(doc, len(doc.text))
    return docs


YAML_HEADER = """# 语料登记表（由 kb/corpus_raw/_scripts/convert.py 生成，请勿手改）
#
# 字段：file / doc_id / name / version / category / source_type / url / note
#       license（许可） / language（zh|en） / tier（1 指南规范、2 权威知识、3 科普问答）
#
# 重新生成：.venv/bin/python kb/corpus_raw/_scripts/convert.py
"""


def write_outputs(docs: list[Doc], out_dir: Path, report: Report, *, dry_run: bool) -> None:
    if dry_run:
        return
    for old in out_dir.rglob("*.md"):
        old.unlink()
    for doc in docs:
        target = out_dir / doc.rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.text + "\n", encoding="utf-8")

    lines = [YAML_HEADER, "files:"]
    for doc in sorted(docs, key=lambda item: item.rel_path):
        lines.append(f"  - file: {doc.rel_path}")
        lines.append(f"    doc_id: {doc.doc_id}")
        lines.append(f"    name: {json.dumps(doc.name, ensure_ascii=False)}")
        lines.append(f"    version: {json.dumps(doc.version, ensure_ascii=False)}")
        lines.append(f"    category: {json.dumps(doc.category, ensure_ascii=False)}")
        lines.append(f"    source_type: {json.dumps(doc.source_type, ensure_ascii=False)}")
        lines.append(f"    url: {json.dumps(doc.url, ensure_ascii=False)}")
        lines.append(f"    note: {json.dumps(doc.note, ensure_ascii=False)}")
        lines.append(f"    license: {json.dumps(doc.license, ensure_ascii=False)}")
        lines.append(f"    language: {doc.language or 'zh'}")
        lines.append(f"    tier: {doc.tier or '3'}")
    (out_dir / "sources.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "n_kept": len(docs),
        "kept_chars": sum(len(doc.text) for doc in docs),
        "n_skipped": len(report.skipped),
        "kept": sorted(report.kept, key=lambda item: item["file"]),
        "skipped": report.skipped,
    }
    (out_dir / "conversion_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # 许可与来源分级清单：对外发布前按这张表决定能不能带出去
    groups: dict[tuple[str, str, str], dict[str, int]] = {}
    for doc in docs:
        key = (doc.tier or "3", doc.license or "未标注", doc.language or "zh")
        item = groups.setdefault(key, {"docs": 0, "chars": 0})
        item["docs"] += 1
        item["chars"] += len(doc.text)
    tier_name = {"1": "一级·指南与政府规范", "2": "二级·权威文献", "3": "三级·科普问答"}
    lines = [
        "# 语料许可与分级清单",
        "",
        "> 由 `kb/corpus_raw/_scripts/convert.py` 生成。对外提交前先按本表过滤：",
        "> 只带「可公开」的许可出库；`kb/index/chunks.jsonl` 每条都带 license/language/tier 字段。",
        "",
        "| 分级 | 许可 | 语言 | 份数 | 字数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for (tier, lic, lang), item in sorted(groups.items()):
        name = tier_name.get(tier, tier)
        lines.append(f"| {name} | {lic} | {lang} | {item['docs']} | {item['chars'] / 1e4:.1f} 万字 |")
    lines += ["", "## 收录口径", "",
              "- 收录：政府文件与地方规范、Europe PMC 开放获取全文（CC BY）、中文医学百科问答（Apache-2.0）。",
              "- 排除：正式测评集的题源池（CMB / CMExam / MedQA / MedMCQA / MMLU / HealthBench /",
              "  AgentClinic / CmedqaRetrieval / MedCalc-Bench），避免评测泄漏；",
              "  医患对话与 SFT 语料不作可引用事实来源；招聘、申请表、专业目录、产业政策等非临床文件在转换阶段剔除。",
              "- 剔除明细见 `conversion_report.json`。"]
    # 放在原始层：kb/corpus 下只允许出现登记过的语料文件，多一份 md 会被建库判为「未登记」
    (RAW / "语料分级与许可.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_chunks(out_dir: Path) -> tuple[int, dict[str, int]]:
    """用 kb 自己的切块器预统计块数（不加载嵌入模型）。"""
    from kb.config import get_config
    from kb.corpus import build_chunks

    chunks, errors = build_chunks(out_dir, get_config())
    by_tier: dict[str, int] = {}
    for chunk in chunks:
        key = chunk.tier or "?"
        by_tier[key] = by_tier.get(key, 0) + 1
    if errors:
        print(f"  [切块警告] {len(errors)} 条：{errors[:3]}")
    return len(chunks), dict(sorted(by_tier.items()))


def dedupe_docs(docs: list[Doc], report: Report) -> list[Doc]:
    """同一份政府文件常出现在多个站点（国务院/部委/地方），按正文指纹去重。"""
    seen: dict[tuple[str, int], str] = {}
    kept: list[Doc] = []
    dropped: set[str] = set()
    for doc in docs:
        fingerprint = re.sub(r"[^\w\u4e00-\u9fff]", "", doc.text)[:1200]
        key = (hashlib.md5(fingerprint.encode("utf-8")).hexdigest(), round(len(doc.text) / 200))
        if key in seen:
            dropped.add(doc.rel_path)
            report.skip(doc.origin, f"正文与已入库文档重复（{seen[key]}）")
            continue
        seen[key] = doc.rel_path
        kept.append(doc)
    if dropped:
        report.kept = [item for item in report.kept if item["file"] not in dropped]
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="corpus_raw → kb/corpus 预处理")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count-chunks", action="store_true")
    parser.add_argument("--baike-target", type=int, default=45000)
    parser.add_argument("--only", default="", help="只跑指定适配器：gov,province,europepmc,baike")
    args = parser.parse_args()

    only = {item.strip() for item in args.only.split(",") if item.strip()}
    report = Report()
    docs: list[Doc] = []
    if not only or "gov" in only:
        docs += adapter_gov_cn(report)
    if not only or "province" in only:
        docs += adapter_province(report)
    if not only or "europepmc" in only:
        docs += adapter_europepmc(report)
    if not only or "baike" in only:
        docs += adapter_baike(report, target=args.baike_target)

    docs = [doc for doc in docs if doc.rel_path and doc.text.strip()]
    docs = dedupe_docs(docs, report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(docs, out_dir, report, dry_run=args.dry_run)

    by_tier: dict[str, int] = {}
    chars: dict[str, int] = {}
    for doc in docs:
        by_tier[doc.tier] = by_tier.get(doc.tier, 0) + 1
        chars[doc.tier] = chars.get(doc.tier, 0) + len(doc.text)
    print(
        f"[convert] 入库 {len(docs)} 份、{sum(chars.values()) / 1e6:.1f}M 字；"
        f"份数按 tier {dict(sorted(by_tier.items()))}，字数按 tier "
        f"{ {k: round(v / 1e6, 1) for k, v in sorted(chars.items())} }"
    )
    print(f"[convert] 剔除 {len(report.skipped)} 份（明细见 conversion_report.json）")
    if args.count_chunks and not args.dry_run:
        n_chunks, tier_chunks = count_chunks(out_dir)
        print(f"[convert] 预计切块 {n_chunks} 条，按 tier {tier_chunks}")
    return 0
def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_text(elem) -> str:
    return norm_text("".join(elem.itertext()))


def normalize_license(text: str) -> str:
    """把期刊里那段许可长句归一成可比较的短标签。"""
    low = (text or "").lower()
    if not low:
        return "CC BY（Europe PMC 仅取开放获取）"
    if "by-nc" in low:
        return "CC BY-NC（不可商用）"
    if "by-sa" in low:
        return "CC BY-SA"
    if "by/4.0" in low or "attribution 4.0" in low or "cc by 4" in low:
        return "CC BY 4.0"
    if "cc by" in low or "cc-by" in low or "creative commons attribution" in low:
        return "CC BY"
    return norm_text(text)[:60]


def adapter_europepmc(report: Report, limit: int = 0) -> list[Doc]:
    """Europe PMC 开放获取全文（CC BY）：正文按 <sec><title> 转标题层级。"""
    import xml.etree.ElementTree as ET

    docs: list[Doc] = []
    root = RAW / "literature_oa" / "europepmc"
    for path in sorted(root.glob("*.xml"))[: limit or None]:
        origin = path.relative_to(RAW).as_posix()
        try:
            article = ET.parse(path).getroot()
        except Exception as exc:  # noqa: BLE001
            report.skip(origin, f"XML 解析失败：{exc}")
            continue

        title = ""
        journal = ""
        license_text = ""
        for elem in article.iter():
            name = _localname(elem.tag)
            if name == "article-title" and not title:
                title = _xml_text(elem)
            elif name == "journal-title" and not journal:
                journal = _xml_text(elem)
            elif name == "license" and not license_text:
                license_text = _xml_text(elem)  # 保留全文，归一化时再截断
        if not title:
            report.skip(origin, "未找到文章标题")
            continue

        lines = [f"# {title}", ""]
        if journal:
            lines += [f"来源期刊：{journal}", ""]
        for elem in article.iter():
            if _localname(elem.tag) == "abstract":
                abstract = _xml_text(elem)
                if abstract:
                    lines += ["## Abstract", "", abstract, ""]
                break
        for body in article.iter():
            if _localname(body.tag) != "body":
                continue
            for sec in body:
                if _localname(sec.tag) != "sec":
                    continue
                sec_title = ""
                paragraphs: list[str] = []
                for child in sec:
                    name = _localname(child.tag)
                    if name == "title":
                        sec_title = _xml_text(child)
                    elif name == "p":
                        paragraphs.append(_xml_text(child))
                    elif name == "sec":
                        sub_title = ""
                        for sub in child:
                            if _localname(sub.tag) == "title":
                                sub_title = _xml_text(sub)
                            elif _localname(sub.tag) == "p":
                                text = _xml_text(sub)
                                paragraphs.append(f"**{sub_title}** {text}" if sub_title else text)
                if sec_title and paragraphs:
                    lines += [f"## {sec_title}", "", "\n\n".join(paragraphs), ""]
            break

        text = norm_text("\n".join(lines))
        if len(text) < 1500:
            report.skip(origin, f"正文过短（{len(text)} 字），可能只有摘要")
            continue
        doc = Doc(
            rel_path=f"literature_oa/europepmc/{slug(title)}_{md_hash(origin)}.md",
            doc_id=f"pmc-{md_hash(origin)}",
            name=title,
            text=text,
            category="开放获取文献",
            source_type="期刊全文（Europe PMC）",
            version="2026-09 采集",
            url="",
            note=f"Europe PMC 开放获取全文；许可：{license_text or 'CC BY'}",
            license=normalize_license(license_text),
            language="en",
            tier="2",
            origin=origin,
        )
        docs.append(doc)
        report.keep(doc, len(doc.text))
    return docs


_BAIKE_KEEP = (
    "病", "症状", "治疗", "诊断", "检查", "用药", "药物", "手术", "护理", "预防",
    "病因", "原因", "鉴别", "并发", "预后", "传染", "感染", "炎", "癌", "瘤",
    "综合征", "中毒", "过敏", "疫苗", "血糖", "血压", "血脂", "尿酸", "贫血",
    "骨折", "结石", "溃疡", "出血", "疼", "痛", "发热", "水肿", "黄疸", "激素",
    "免疫", "遗传", "畸形", "阻塞", "衰竭", "梗死", "风湿", "甲状腺", "肝", "肾",
    "肺", "胃", "肠", "心", "脑", "胆", "胰", "脾", "眼", "耳", "鼻", "咽喉",
    "皮肤", "口腔", "牙", "关节", "脊柱", "前列", "子宫", "卵巢",
)
_BAIKE_DROP = re.compile(
    r"形态特征|栽培|养殖|饲养|花语|寓意|风水|算命|星座|生肖|旅游|景点|电影|歌曲|"
    r"明星|游戏|手机|汽车|价格|多少钱|哪里买|加盟|代理|招聘|考试|证书|户口|报销|"
    r"挂号费|医院简介|科室介绍|医生简介|丰胸|增高|美白|祛斑|减肥药|化妆品|香水|"
    r"彩票|股票|装修|婚礼|宠物|植物|花草|蔬菜|水果|食谱|做法|好吃"
)
_BAIKE_QUESTION_RE = re.compile(r"^(.{4,60}?)[?？]")


def adapter_baike(report: Report, target: int = 45000) -> list[Doc]:
    """中文医学百科问答精选：先筛医学相关性，再按信息量取前 target 条。"""
    origin = "hf_datasets/medical-shibing624/train_encyclopedia.json"
    path = RAW / origin
    if not path.is_file():
        report.skip(origin, "文件不存在")
        return []

    seen: set[str] = set()
    candidates: list[tuple[int, str, str]] = []
    total = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total += 1
            try:
                text = norm_text((json.loads(line) or {}).get("text") or "")
            except json.JSONDecodeError:
                continue
            if not 80 <= len(text) <= 2000:
                continue
            match = _BAIKE_QUESTION_RE.match(text)
            if not match:
                continue
            question = match.group(1) + "？"
            if _BAIKE_DROP.search(question[:40]):
                continue
            body = text[len(question) :].strip()
            hits = sum(1 for kw in _BAIKE_KEEP if kw in question)
            if hits < 1:
                continue
            if hits < 2 and sum(1 for kw in _BAIKE_KEEP if kw in body[:200]) < 2:
                continue
            key = re.sub(r"[^\w\u4e00-\u9fff]", "", question)
            if key in seen:
                continue
            seen.add(key)
            score = hits * 10 + min(len(body), 800) // 100
            candidates.append((score, question, body))

    candidates.sort(key=lambda item: -item[0])
    picked = candidates[:target]
    report.skip(
        origin,
        f"百科共 {total} 条：医学筛选命中 {len(candidates)} 条，入库 {len(picked)} 条",
    )

    buckets: dict[int, list[str]] = {}
    for _score, question, body in picked:
        bucket = int(md_hash(question)[:4], 16) % 48
        buckets.setdefault(bucket, []).append(f"【{question}】\n{body}")

    docs: list[Doc] = []
    for bucket in sorted(buckets):
        text = norm_text(f"# 中文医学百科问答 · 第{bucket + 1}卷\n\n" + "\n\n".join(buckets[bucket]))
        doc = Doc(
            rel_path=f"encyclopedia/baike_{bucket:02d}.md",
            doc_id=f"baike-{bucket:02d}",
            name=f"中文医学百科问答（第{bucket + 1}卷）",
            text=text,
            category="医学百科",
            source_type="百科/问答（科普级）",
            version="2026-09 筛选",
            url="https://huggingface.co/datasets/shibing624/medical",
            note="来源 shibing624/medical（Apache-2.0），医学关键词筛选 + 去重",
            license="Apache-2.0",
            language="zh",
            tier="3",
            origin=origin,
        )
        docs.append(doc)
        report.keep(doc, len(doc.text))
    return docs


def adapter_province(report: Report) -> list[Doc]:
    """省级卫健委 / 疾控中心：html 正文与 docx/pdf 附件。"""
    docs: list[Doc] = []
    root = RAW / "guidelines_cn" / "province"
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        origin = path.relative_to(RAW).as_posix()
        if path.suffix.lower() not in (".html", ".docx", ".pdf"):
            report.skip(origin, "格式暂不转换（.doc/.xlsx 需额外工具且临床价值低）")
            continue
        name_noise = _NOISE_TITLE.search(path.stem)
        if name_noise:
            report.skip(origin, f"文件名命中非医学模式：{name_noise.group(0)}")
            continue
        try:
            body, title = read_any(path)
        except Exception as exc:  # noqa: BLE001
            report.skip(origin, f"读取失败：{type(exc).__name__}: {exc}")
            continue
        if not body:
            reason = (
                "未定位到正文容器（列表/导航页，非文章正文）"
                if path.suffix.lower() == ".html"
                else "解析后无正文"
            )
            report.skip(origin, reason)
            continue
        if not title:
            title = body.split("\n", 1)[0][:60]
        ok, why = is_medical(title, body, check_body_head=True)
        if not ok:
            report.skip(origin, why)
            continue
        province = path.parent.name
        doc = build_md_doc(
            rel_path=f"guidelines_cn/province/{slug(title)}_{md_hash(origin)}.md",
            doc_id=f"prov-{md_hash(origin)}",
            title=title,
            body=body,
            category=f"地方规范（{province}）",
            source_type="政府/事业单位公开文件",
            version="2026-09 采集",
            url="",
            note=f"来源：{province}，正文由 html/docx/pdf 转换",
            license_text="公开文件（可公开，注明出处）",
            language="zh",
            tier="1",
            origin=origin,
        )
        docs.append(doc)
        report.keep(doc, len(doc.text))
    return docs


if __name__ == "__main__":
    raise SystemExit(main())
