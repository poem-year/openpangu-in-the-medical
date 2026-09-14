"""BM25 词法检索（中文按单字 + 双字切，英文/数字按词切）。

上一个版本用 `query.split()` 做分词，中文等于没分词，BM25 那一支实际不起作用。
"""

from __future__ import annotations

import re
from typing import Sequence

TOKENIZER_VERSION = "zh-bigram-v2"

_RUN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文：以双字片段为主，单字只在整段只有一个字时保留；英文/数字：小写整词。

    为什么去掉单字：单字分词会把「病」「症」「炎」这类几乎每篇都有的字当特征，
    实测查询「川崎病」时 BM25 的第一名是「狂犬病」——两者只共享一个「病」字。
    双字（川崎 / 崎病、高血 / 血压）才有区分度，去掉单字后词法那一路才真的有信息量。
    """
    tokens: list[str] = []
    for run in _RUN_RE.findall(text or ""):
        if "\u4e00" <= run[0] <= "\u9fff":
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run.lower())
    return tokens


class Bm25Index:
    """基于 rank_bm25 的词法索引；token 由正文重建，不额外落盘。"""

    def __init__(self, texts: Sequence[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.tokenized = [tokenize(text) for text in texts]
        self._model = BM25Okapi(self.tokenized) if self.tokenized else None

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """返回 (行号, BM25 分数)，按分数降序；分数为 0 的不返回。"""
        if self._model is None or k <= 0:
            return []
        scores = self._model.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in order if scores[i] > 0]
