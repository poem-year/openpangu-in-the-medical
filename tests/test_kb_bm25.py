"""中文 BM25 分词与区分度测试（D2 回归）。

背景：原来的单字 + 双字分词把「病/症/炎」这类常见字当特征，实测查询「川崎病」
时 BM25 的第一名是「狂犬病」（只共享一个「病」字），词法那一路反而拖累混合检索。
"""

from __future__ import annotations

from kb.bm25 import Bm25Index, tokenize


class TestTokenizer:
    def test_bigram_only_for_multi_char_runs(self):
        assert tokenize("高血压") == ["高血", "血压"]

    def test_no_single_char_noise(self):
        tokens = tokenize("川崎病")
        assert "病" not in tokens
        assert tokens == ["川崎", "崎病"]

    def test_single_char_run_is_kept(self):
        assert tokenize("痛") == ["痛"]

    def test_mixed_ascii_and_chinese(self):
        assert tokenize("CT 检查") == ["ct", "检查"]

    def test_empty_text(self):
        assert tokenize("") == []


class TestDiscrimination:
    def _corpus(self):
        # 20 条语料，保证 IDF 正常；只有第 0 条含「川崎」「崎病」两个双字
        texts = ["【川崎病】定义：儿童急性血管炎。"] + [
            f"【疾病{i}】定义：某种疾病，症状与治疗如下。" for i in range(1, 20)
        ]
        return texts

    def test_rare_disease_name_wins_over_shared_char(self):
        texts = self._corpus()
        index = Bm25Index(texts)
        top = index.top("川崎病", 3)
        assert top, "应该能召回"
        assert top[0][0] == 0, "含「川崎/崎病」双字的条目必须排第一"

    def test_unrelated_disease_is_not_matched(self):
        texts = self._corpus()
        index = Bm25Index(texts)
        hits = dict(index.top("川崎病", 5))
        assert 1 not in hits, "只共享「病」字的条目不应被词法命中"

    def test_shared_bigram_matches(self):
        texts = ["【高血压】限盐。", "【糖尿病】控糖。"] + [f"【无关{i}】内容。" for i in range(18)]
        index = Bm25Index(texts)
        assert index.top("高血压", 1)[0][0] == 0
