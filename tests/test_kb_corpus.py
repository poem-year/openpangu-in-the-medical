"""kb 语料层测试：切块规则、出处（section）、chunk_id 稳定性、登记校验。"""

from __future__ import annotations

import pytest
import yaml

from kb.config import KBConfig
from kb.corpus import (
    CorpusError,
    build_chunks,
    load_sources,
    make_chunk_id,
    read_document,
    split_document,
)


def _split(text: str, default_section: str = "心血管内科", **kwargs):
    return split_document(
        text, doc_id="cardio", default_section=default_section, config=KBConfig(**kwargs)
    )


def _write_corpus(tmp_path, files: dict[str, str], entries: list[dict]):
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        path = corpus / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (corpus / "sources.yaml").write_text(
        yaml.safe_dump({"files": entries}, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return corpus


class TestSplitting:
    def test_markdown_headings_become_section_path(self):
        text = (
            "# 心血管内科\n\n"
            "## 高血压\n\n【高血压】\n定义：诊室血压非同日三次≥140/90mmHg。\n\n"
            "## 冠心病\n\n【冠心病】\n症状：活动时胸骨后压榨样疼痛。\n"
        )
        chunks = _split(text)
        assert [chunk.section for chunk in chunks] == [
            "心血管内科 / 高血压",
            "心血管内科 / 冠心病",
        ]
        assert [chunk.seq for chunk in chunks] == [1, 2]

    def test_entry_markers_used_when_no_headings(self):
        text = "【高血压】\n定义：血压升高。\n\n【冠心病】\n症状：胸痛。\n"
        chunks = _split(text)
        assert [chunk.section for chunk in chunks] == ["高血压", "冠心病"]

    def test_paragraph_aggregation_falls_back_to_default_section(self):
        text = "\n\n".join(f"第{i}段：" + "内容" * 30 for i in range(6))
        chunks = _split(text, default_section="某指南")
        assert chunks, "段落模式至少要产出一块"
        assert {chunk.section for chunk in chunks} == {"某指南"}

    def test_long_block_is_split_and_keeps_section(self):
        body = "。".join(f"第{i}句" + "细节" * 20 for i in range(40))
        chunks = _split(f"# 总论\n\n## 细则\n\n{body}\n")
        assert len(chunks) > 1, "超过 chunk_split_size 的块必须被切开"
        assert {chunk.section for chunk in chunks} == {"总论 / 细则"}
        assert all(len(chunk.text) <= 700 for chunk in chunks)

    def test_headings_then_entries_keeps_one_disease_per_chunk(self):
        """回归：`# 科室` + 多条【疾病】不能被压成一块（原来的高危 bug）。"""
        text = (
            "# 心血管内科\n\n"
            "【高血压】\n定义：血压升高。\n\n"
            "【冠心病】\n症状：胸痛。\n\n"
            "【心衰】\n症状：气促。\n"
        )
        chunks = _split(text, default_section="心血管内科")
        assert len(chunks) == 3, f"应切出 3 个病种，实际 {len(chunks)} 块"
        assert [chunk.section for chunk in chunks] == [
            "心血管内科 / 高血压",
            "心血管内科 / 冠心病",
            "心血管内科 / 心衰",
        ]
        # 【条目名】同时进 section 和正文首行：正文必须能自我说明是哪一条，
        # 否则用病名查询时既匹配不上向量也匹配不上词法（实测 41 个病名排不到第一，40 个是这原因）
        assert [chunk.text for chunk in chunks] == [
            "【高血压】\n定义：血压升高。",
            "【冠心病】\n症状：胸痛。",
            "【心衰】\n症状：气促。",
        ]

    def test_heading_name_equal_to_entry_name_is_not_duplicated(self):
        text = "# 心血管内科\n\n## 高血压\n\n【高血压】\n定义：血压升高。\n"
        chunks = _split(text, default_section="心血管内科")
        assert [chunk.section for chunk in chunks] == ["心血管内科 / 高血压"]

    def test_entries_under_second_level_heading(self):
        text = "# 内科\n\n## 心血管\n\n【高血压】\n定义：血压升高。\n\n【冠心病】\n症状：胸痛。\n"
        chunks = _split(text, default_section="内科")
        assert [chunk.section for chunk in chunks] == ["内科 / 心血管 / 高血压", "内科 / 心血管 / 冠心病"]

    def test_heading_with_preamble_content_keeps_default_section(self):
        text = "（本文件为整理稿）\n\n# 总论\n\n【概述】\n整体说明。\n"
        chunks = _split(text, default_section="某指南")
        assert [chunk.section for chunk in chunks] == ["某指南", "总论 / 概述"]

    def test_duplicate_text_gets_distinct_chunk_ids(self):
        """回归：同一文档内出现完全相同的条目时，chunk_id 必须仍唯一。"""
        text = "【高血压】\n相同正文。\n\n【高血压】\n相同正文。\n"
        chunks = _split(text)
        ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunks) == 2 and len(set(ids)) == 2, f"ID 撞了：{ids}"

    def test_heading_sections_keep_title_in_body(self):
        """标题模式下正文也要带上小节名（标题名本身是检索线索）。"""
        chunks = _split("# 总论\n\n## 诊断要点\n\n需要结合影像学检查。\n")
        assert [chunk.section for chunk in chunks] == ["总论 / 诊断要点"]
        assert chunks[0].text.startswith("诊断要点")


class TestChunkId:
    def test_stable_and_content_bound(self):
        first = make_chunk_id("cardio", "同样的正文")
        second = make_chunk_id("cardio", "同样的正文")
        changed = make_chunk_id("cardio", "同样的正支")
        assert first == second
        assert first != changed
        assert first.startswith("kb::cardio::")
        assert len(first) == len("kb::cardio::") + 12

    def test_doc_id_participates(self):
        assert make_chunk_id("a", "正文") != make_chunk_id("b", "正文")


class TestRegistry:
    def test_missing_sources_yaml(self, tmp_path):
        with pytest.raises(CorpusError, match="sources.yaml"):
            load_sources(tmp_path)

    def test_empty_files_list(self, tmp_path):
        (tmp_path / "sources.yaml").write_text("files: []\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="至少含一条"):
            load_sources(tmp_path)

    def test_duplicate_doc_id_is_rejected(self, tmp_path):
        """doc_id 决定 chunk_id 前缀，两份文件共用一个 doc_id 必须报错。"""
        corpus = _write_corpus(
            tmp_path,
            {"a.md": "# 甲\n\n内容甲。\n", "b.md": "# 乙\n\n内容乙。\n"},
            [{"file": "a.md", "doc_id": "same"}, {"file": "b.md", "doc_id": "same"}],
        )
        with pytest.raises(CorpusError, match="重复使用"):
            load_sources(corpus)

    def test_registered_but_missing_file(self, tmp_path):
        corpus = _write_corpus(
            tmp_path,
            {"存在.md": "# 标题\n\n正文。\n"},
            [{"file": "存在.md", "doc_id": "a"}, {"file": "不存在.md", "doc_id": "b"}],
        )
        chunks, errors = build_chunks(corpus, KBConfig())
        assert chunks, "已存在的语料照常入库"
        assert any("不存在.md" in message and "找不到" in message for message in errors)

    def test_unregistered_file_is_reported(self, tmp_path):
        corpus = _write_corpus(
            tmp_path,
            {"已登记.md": "# 标题\n\n正文。\n", "没登记.md": "# 标题\n\n正文。\n"},
            [{"file": "已登记.md", "doc_id": "a"}],
        )
        _, errors = build_chunks(corpus, KBConfig())
        assert any("没登记.md" in message and "未在 sources.yaml 登记" in message for message in errors)

    def test_source_fields_carried_into_chunks(self, tmp_path):
        corpus = _write_corpus(
            tmp_path,
            {"指南.md": "# 高血压\n\n限盐每日 5 克。\n"},
            [
                {
                    "file": "指南.md",
                    "doc_id": "htn",
                    "name": "中国高血压防治指南",
                    "version": "2024版",
                    "category": "心血管内科",
                }
            ],
        )
        chunks, errors = build_chunks(corpus, KBConfig())
        assert not errors
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.source_name == "中国高血压防治指南"
        assert chunk.source_version == "2024版"
        assert chunk.section == "高血压"
        assert chunk.category == "心血管内科"
        assert chunk.file_sha1


class TestReadDocument:
    def test_empty_text_file_reports_clearly(self, tmp_path):
        path = tmp_path / "空.md"
        path.write_text("   \n\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="没有任何文本"):
            read_document(path)

    def test_reads_utf8(self, tmp_path):
        path = tmp_path / "中文.txt"
        path.write_text("高血压诊疗要点。", encoding="utf-8")
        assert "高血压" in read_document(path)
