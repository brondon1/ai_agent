"""文本向量化。默认使用 FastEmbed 在本地运行，无需额外的 API。"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """首次使用时会从 Hugging Face 下载模型（bge-small-zh 约 100MB）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        from fastembed import TextEmbedding  # 延迟导入，避免未使用 RAG 时加载 onnxruntime

        self._model = TextEmbedding(model_name)
        self.dim = TextEmbedding.get_embedding_size(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]
