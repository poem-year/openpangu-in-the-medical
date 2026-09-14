"""嵌入后端：真实后端走 BGE-M3，测试用假后端（确定性、不加载模型）。"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化；零向量保持为零，避免除零。"""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class EmbeddingBackend(Protocol):
    """嵌入后端协议：文本 → L2 归一化向量。"""

    model_name: str
    dim: int

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class BgeM3Backend:
    """BGE-M3（或任意 HuggingFace 句向量模型）后端，CPU 运行。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str = "cpu",
        batch_size: int = 32,
        query_prompt: str = "",
        doc_prompt: str = "",
    ) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name
        encode_kwargs: dict = {"normalize_embeddings": True, "batch_size": batch_size}
        if doc_prompt:
            encode_kwargs["prompt"] = doc_prompt
        kwargs: dict = {
            "model_name": model_name,
            "model_kwargs": {"device": device},
            "encode_kwargs": encode_kwargs,
        }
        if query_prompt:
            kwargs["query_encode_kwargs"] = {
                "normalize_embeddings": True,
                "prompt": query_prompt,
            }
        self._embeddings = HuggingFaceEmbeddings(**kwargs)
        self._dim: int | None = None

    @property
    def dim(self) -> int:  # type: ignore[override]
        if self._dim is None:
            client = getattr(self._embeddings, "client", None)
            dim = None
            if client is not None and hasattr(client, "get_sentence_embedding_dimension"):
                dim = client.get_sentence_embedding_dimension()
            self._dim = int(dim) if dim else len(self._embeddings.embed_query("维度探测"))
        return self._dim

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._embeddings.embed_documents(list(texts))
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def encode_query(self, text: str) -> np.ndarray:
        vector = np.asarray(self._embeddings.embed_query(text), dtype=np.float32)
        return l2_normalize(vector)[0]


class FakeBackend:
    """测试用假后端。

    传了 `vectors` 时按文本精确查表（便于断言排序）；查不到时用
    文本哈希生成确定性向量，保证同一文本永远得到同一向量。
    """

    model_name = "fake"

    def __init__(self, vectors: dict[str, Sequence[float]] | None = None, dim: int = 8) -> None:
        self._vectors = {k: l2_normalize(np.asarray(v, dtype=np.float32))[0] for k, v in (vectors or {}).items()}
        self._dim = dim if not self._vectors else len(next(iter(self._vectors.values())))

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self._dim

    def _one(self, text: str) -> np.ndarray:
        if text in self._vectors:
            return self._vectors[text]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [((digest[i % len(digest)] / 255.0) - 0.5) for i in range(self.dim)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return np.asarray([value / norm for value in raw], dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.vstack([self._one(text) for text in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return self._one(text)


class NpuBackend:
    """昇腾 NPU 建库后端（只在建库时用；查询侧仍走 CPU 的 BgeM3Backend）。

    为什么单独一个实现：这台机器 128 核 CPU 跑 BGE-M3 只有约 5~6 块/秒，
    8.4 万块要 4 小时以上；NPU 上同一个模型是几十倍吞吐。向量口径保持一致
    （mean pooling + L2 归一化，与 BGE-M3 官方句向量用法相同），
    所以查询向量用 CPU 算、库向量用 NPU 算，余弦仍然可比。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str = "npu:0",
        batch_size: int = 64,
        max_length: int = 1024,
        query_prompt: str = "",
        doc_prompt: str = "",
    ) -> None:
        import torch
        import torch_npu  # noqa: F401 - 注册 NPU 后端
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = (
            AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
            .to(device)
            .eval()
        )
        hidden = getattr(self._model.config, "hidden_size", None)
        self._dim: int | None = int(hidden) if hidden else None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode_documents(["维度探测"]).shape[1])
        return self._dim

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = self._tokenizer(
                list(texts[start : start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self._model(**batch).last_hidden_state
            # BGE-M3 的句向量取 [CLS] 位（与 sentence-transformers 的
            # 1_Pooling/config.json 一致）；用 mean pooling 会和查询侧对不上，
            # 实测同一句话余弦只有 0.79。
            pooled = hidden[:, 0]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.float().cpu().numpy())
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(self._encode(texts))

    def encode_query(self, text: str) -> np.ndarray:
        vector = self._encode([text])[0]
        return l2_normalize(vector)[0]


def create_backend(config, **kwargs) -> EmbeddingBackend:
    """按 KB_DEVICE 选后端：npu* → NpuBackend，其余 → BgeM3Backend。"""
    device = (config.device or "cpu").lower()
    if device.startswith("npu"):
        return NpuBackend(
            config.embed_model,
            device=device,
            batch_size=config.embed_batch_size,
            query_prompt=config.query_prompt,
            doc_prompt=config.doc_prompt,
        )
    return BgeM3Backend(
        config.embed_model,
        device=config.device,
        batch_size=config.embed_batch_size,
        query_prompt=config.query_prompt,
        doc_prompt=config.doc_prompt,
    )
