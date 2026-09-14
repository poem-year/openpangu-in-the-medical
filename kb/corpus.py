"""语料接入：sources.yaml 登记 + md/txt/pdf 读取 + 结构化切块。

设计原则（见 `kb/README.md`）：
- 出处只来自 `sources.yaml` 与文档标题层级，**不从正文猜出处**；
- 一病种 / 一小节一块，不跨小节合并（上一个版本就是把两个病种切进同一块）；
- chunk_id 由内容决定：`kb::<doc_id>::<内容哈希前12位>`，内容变则 ID 变。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from kb.config import KBConfig

SOURCES_FILENAME = "sources.yaml"
SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")
# 切块逻辑的版本号：写进索引 manifest；改了切块规则就升版本，建库时会强制全量重建。
CHUNKER_VERSION = "v2-entry-prefix"

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_ENTRY_RE = re.compile(r"^\s*【([^】]{1,60})】\s*$", re.MULTILINE)


class CorpusError(RuntimeError):
    """语料层错误（登记缺失、文件读不出、解析为空等）。"""


@dataclass(frozen=True)
class SourceEntry:
    """sources.yaml 里的一条语料登记。"""

    file: str
    doc_id: str
    name: str = ""
    version: str = ""
    category: str = ""
    source_type: str = ""
    url: str = ""
    note: str = ""
    # 以下三项供来源分级与对外发布过滤用（只增不改，缺省为空字符串）：
    license: str = ""
    language: str = ""
    tier: str = ""


@dataclass(frozen=True)
class CorpusFile:
    """扫描到的一份语料文件。"""

    path: Path
    rel_path: str
    sha1: str
    entry: SourceEntry


@dataclass(frozen=True)
class CorpusChunk:
    """切块结果，字段即索引记录字段。"""

    chunk_id: str
    doc_id: str
    text: str
    section: str
    seq: int
    source_name: str
    source_version: str
    category: str
    file_sha1: str
    rel_path: str
    license: str = ""
    language: str = ""
    tier: str = ""


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_chunk_id(doc_id: str, text: str, occurrence: int = 1) -> str:
    """内容决定 ID：同一片段多次建库必须得到同一 ID，改一个字符则 ID 变。

    `occurrence` 用于同一文档内出现多段完全相同的正文时消歧：
    第 1 段保持 `kb::<doc_id>::<哈希>` 的既有格式，第 2 段起在后面加 `#N`，
    保证 chunk_id 在整个知识库里唯一（接口规范 §2.2 要求）。
    """
    raw = f"{doc_id}||{text}" if occurrence <= 1 else f"{doc_id}||{text}||#{occurrence}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return f"kb::{doc_id}::{h[:12]}"


# --------------------------------------------------------------------------
# sources.yaml
# --------------------------------------------------------------------------
def load_sources(corpus_dir: Path) -> dict[str, SourceEntry]:
    """读 sources.yaml，返回 {相对路径: SourceEntry}。文件缺失直接报错。"""
    import yaml

    path = corpus_dir / SOURCES_FILENAME
    if not path.is_file():
        raise CorpusError(
            f"缺少语料登记文件：{path}\n"
            f"请在 {SOURCES_FILENAME} 的 `files:` 下登记每份语料的 "
            "file/doc_id/name/version/category/source_type（模板见 kb/corpus/sources.yaml）"
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - 只在人工改坏时触发
        raise CorpusError(f"{path} 不是合法 YAML：{exc}") from exc

    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise CorpusError(f"{path} 的 `files:` 必须是至少含一条记录的列表")

    sources: dict[str, SourceEntry] = {}
    doc_id_owner: dict[str, str] = {}
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, dict):
            raise CorpusError(f"{path} 第 {index + 1} 条登记不是映射（应为 - file: ... 形式）")
        rel = str(raw.get("file") or "").strip()
        doc_id = str(raw.get("doc_id") or "").strip()
        if not rel:
            raise CorpusError(f"{path} 第 {index + 1} 条登记缺 `file`")
        if not doc_id:
            raise CorpusError(f"{path} 第 {index + 1} 条登记（{rel}）缺 `doc_id`")
        if rel in sources:
            raise CorpusError(f"{path} 里 {rel} 重复登记")
        if doc_id in doc_id_owner:
            raise CorpusError(
                f"{path} 里 doc_id {doc_id!r} 被 {doc_id_owner[doc_id]} 与 {rel} 重复使用；"
                "doc_id 决定 chunk_id 前缀，必须全局唯一"
            )
        doc_id_owner[doc_id] = rel
        sources[rel] = SourceEntry(
            file=rel,
            doc_id=doc_id,
            name=str(raw.get("name") or Path(rel).stem).strip(),
            version=str(raw.get("version") or "").strip(),
            category=str(raw.get("category") or "").strip(),
            source_type=str(raw.get("source_type") or "").strip(),
            url=str(raw.get("url") or "").strip(),
            note=str(raw.get("note") or "").strip(),
            license=str(raw.get("license") or "").strip(),
            language=str(raw.get("language") or "").strip(),
            tier=str(raw.get("tier") or "").strip(),
        )
    return sources


def scan_corpus(corpus_dir: Path) -> tuple[list[CorpusFile], list[str]]:
    """扫描语料目录。返回 (已登记且存在的文件, 错误清单)。"""
    sources = load_sources(corpus_dir)
    errors: list[str] = []
    found: list[CorpusFile] = []

    for rel, entry in sorted(sources.items()):
        path = corpus_dir / rel
        if not path.is_file():
            errors.append(f"sources.yaml 登记了 {rel}，但语料目录里找不到该文件")
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            errors.append(f"{rel} 的扩展名不在支持范围 {SUPPORTED_SUFFIXES} 内")
            continue
        found.append(CorpusFile(path=path, rel_path=rel, sha1=sha1_file(path), entry=entry))

    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.name == SOURCES_FILENAME:
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel = path.relative_to(corpus_dir).as_posix()
        if rel not in sources:
            errors.append(f"语料文件 {rel} 未在 sources.yaml 登记（出处无法追溯，已跳过）")

    return found, errors


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def read_document(path: Path) -> str:
    """读 md/txt/pdf 为纯文本；解析为空时报错，不静默放过。"""
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001 - pypdf 的异常类型不稳定
            raise CorpusError(f"{path.name} 解析失败：{type(exc).__name__}: {exc}") from exc
    else:  # pragma: no cover - scan_corpus 已挡掉
        raise CorpusError(f"{path.name} 不是支持的语料格式")

    if not text.strip():
        raise CorpusError(
            f"{path.name} 解析后没有任何文本（扫描件/图片型 PDF 本模块不支持，"
            "请先转成文本再入库）"
        )
    return text


# --------------------------------------------------------------------------
# 切块
# --------------------------------------------------------------------------
def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题层级切；section 为标题路径，如「心血管内科 / 高血压 / 诊断」。"""
    stack: list[tuple[int, str]] = []
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    current = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            blocks.append((current, body))
        buf.clear()

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = " / ".join(item for _, item in stack)
        else:
            buf.append(line)
    flush()
    return blocks


def _split_by_entries(text: str, default_section: str) -> list[tuple[str, str, str]]:
    """按【条目】切。返回 [(section, prefix, body)]。

    `prefix` 是 `【条目名】`，会留在切块正文第一行。为什么必须留着：病名一旦只进
    section，正文就失去了自我标识——实测用病名查询自己，201 个里 41 个排不到第一，
    其中 40 个的正文里根本没有自己的病名（BM25 也只剩「病/症/炎」这类单字噪声）。
    """
    blocks: list[tuple[str, str, str]] = []
    current: str | None = None
    buf: list[str] = []
    preamble: list[str] = []

    def flush(name: str | None) -> None:
        body = "\n".join(buf).strip()
        if body:
            blocks.append((name or "", f"【{name}】" if name else "", body))
        buf.clear()

    for line in text.splitlines():
        match = _ENTRY_RE.match(line)
        if match:
            if current is None:
                preamble = list(buf)
                buf.clear()
            else:
                flush(current)
            current = match.group(1).strip()
        else:
            buf.append(line)
    if current is None:
        body = "\n".join(buf).strip()
        if body:
            blocks.append((default_section, "", body))
        return blocks
    flush(current)
    if preamble and any(item.strip() for item in preamble):
        blocks.insert(0, (default_section, "", "\n".join(preamble).strip()))
    return blocks


def _split_by_paragraphs(text: str, target_max: int) -> list[tuple[str, str]]:
    """无标题也无条目标记：按段落聚合到目标长度。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks: list[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= target_max:
            current = f"{current}\n\n{para}"
        else:
            blocks.append(current)
            current = para
    if current:
        blocks.append(current)
    return [("", block) for block in blocks]


def _split_long_block(text: str, config: KBConfig) -> list[str]:
    """超长块再切一次，避免单块超出接口规范建议的 50~500 字太多。"""
    if len(text) <= config.chunk_split_size:
        return [text]
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_split_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    parts = [part.strip() for part in splitter.split_text(text) if part.strip()]
    return parts or [text]


def _heading_leaf(path: str) -> str:
    """取标题路径的最后一级，作为该段正文的前缀（标题名本身也是检索线索）。"""
    return path.split(" / ")[-1] if path else ""


def _split_structured(text: str, default_section: str, config: KBConfig) -> list[tuple[str, str, str]]:
    """切块优先级：标题分段 → 段内再按【条目】切；两者都没有才按段落聚合。

    返回 [(section, prefix, body)]：prefix 会留在正文首行当检索与阅读锚点。

    为什么不是「有标题就只用标题」：语料常见的形态是 `# 科室` 下面挂一堆
    `【疾病】` 条目，只按标题切会把整个科室压成一块（实测 10 个病种并成 1 块），
    引用就只能指到一大坨内容，一病种一块的粒度就没了。
    """
    if _HEADING_RE.search(text):
        blocks: list[tuple[str, str, str]] = []
        for path, body in _split_by_headings(text):
            base = path or default_section
            if _ENTRY_RE.search(body):
                for entry_name, entry_prefix, entry_body in _split_by_entries(body, ""):
                    # 标题名与条目名相同时不要拼成「心血管内科 / 高血压 / 高血压」
                    last = base.split(" / ")[-1] if base else ""
                    section = base if (not entry_name or entry_name == last) else f"{base} / {entry_name}"
                    blocks.append((section.strip(" /"), entry_prefix, entry_body))
            else:
                blocks.append((base, _heading_leaf(path), body))
        return blocks
    if _ENTRY_RE.search(text):
        return _split_by_entries(text, default_section)
    return [(section, "", body) for section, body in _split_by_paragraphs(text, config.chunk_target_max)]


def split_document(text: str, *, doc_id: str, default_section: str, config: KBConfig) -> list[CorpusChunk]:
    """把一份文档切成块。返回块的 section/正文，chunk_id 由调用方补齐前缀信息。"""
    raw_blocks = _split_structured(text, default_section, config)
    chunks: list[CorpusChunk] = []
    seq = 0
    occurrence: dict[str, int] = {}
    for section, prefix, body in raw_blocks:
        resolved = section or default_section
        for part in _split_long_block(body, config):
            seq += 1
            # 前缀必须逐块带上：长块切开后，每一块都要能自己说明「这是哪一条」
            text_with_prefix = f"{prefix}\n{part}" if prefix and not part.startswith(prefix) else part
            occurrence[text_with_prefix] = occurrence.get(text_with_prefix, 0) + 1
            chunks.append(
                CorpusChunk(
                    chunk_id=make_chunk_id(doc_id, text_with_prefix, occurrence[text_with_prefix]),
                    doc_id=doc_id,
                    text=text_with_prefix,
                    section=resolved,
                    seq=seq,
                    source_name="",
                    source_version="",
                    category="",
                    file_sha1="",
                    rel_path="",
                )
            )
    return chunks


def build_chunks(corpus_dir: Path, config: KBConfig) -> tuple[list[CorpusChunk], list[str]]:
    """扫描语料并切块。返回 (块列表, 错误清单)；错误非空时调用方应提示并不写索引。"""
    files, errors = scan_corpus(corpus_dir)
    chunks: list[CorpusChunk] = []
    for item in files:
        try:
            text = read_document(item.path)
        except CorpusError as exc:
            errors.append(str(exc))
            continue
        entry = item.entry
        default_section = entry.category or entry.name or Path(item.rel_path).stem
        for chunk in split_document(
            text, doc_id=entry.doc_id, default_section=default_section, config=config
        ):
            chunks.append(
                CorpusChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    section=chunk.section,
                    seq=chunk.seq,
                    source_name=entry.name or Path(item.rel_path).stem,
                    source_version=entry.version,
                    category=entry.category,
                    file_sha1=item.sha1,
                    rel_path=item.rel_path,
                    license=entry.license,
                    language=entry.language,
                    tier=entry.tier,
                )
            )
    return chunks, errors
