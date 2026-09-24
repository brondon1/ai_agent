"""知识库：文本切块、写入 Qdrant、相似度检索，并包装成 Agent 可调用的工具。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ai_agent.rag.embeddings import Embedder
from ai_agent.tools import Tool


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    score: float


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """按段落合并成不超过 size 个字符的块；超长段落按固定窗口切分，块之间保留 overlap 重叠。"""
    if overlap >= size:
        raise ValueError("overlap 必须小于 size")
    chunks: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}" if current else paragraph
            continue
        if current:
            chunks.append(current)
        while len(paragraph) > size:
            chunks.append(paragraph[:size])
            paragraph = paragraph[size - overlap :]
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


class KnowledgeBase:
    def __init__(
        self,
        embedder: Embedder,
        client: QdrantClient | None = None,
        collection: str = "knowledge",
        top_k: int = 4,
    ) -> None:
        self.embedder = embedder
        # 未指定时使用内存模式；生产环境传入 QdrantClient(url="http://localhost:6333")
        self.client = client or QdrantClient(":memory:")
        self.collection = collection
        self.top_k = top_k
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
            )

    def add_texts(self, texts: list[str], source: str = "inline") -> int:
        chunks = [chunk for text in texts for chunk in chunk_text(text)]
        if not chunks:
            return 0
        vectors = self.embedder.embed(chunks)
        self.client.upsert(
            self.collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()), vector=vector, payload={"text": chunk, "source": source}
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
        return len(chunks)

    def add_file(self, path: str | Path) -> int:
        path = Path(path)
        return self.add_texts([path.read_text(encoding="utf-8")], source=str(path))

    def add_path(self, path: str | Path, suffixes: tuple[str, ...] = (".md", ".txt")) -> int:
        """导入单个文件，或目录下所有指定后缀的文件。返回写入的块数。"""
        path = Path(path)
        if path.is_file():
            return self.add_file(path)
        return sum(self.add_file(p) for p in sorted(path.rglob("*")) if p.suffix in suffixes)

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        [vector] = self.embedder.embed([query])
        points = self.client.query_points(
            self.collection, query=vector, limit=k or self.top_k, with_payload=True
        ).points
        return [Hit(p.payload["text"], p.payload["source"], p.score) for p in points]

    def as_tool(self) -> Tool:
        def search_knowledge_base(query: str) -> str:
            hits = self.search(query)
            if not hits:
                return "知识库中没有找到相关内容。"
            return "\n\n".join(
                f"[{i}] 来源：{hit.source}（相似度 {hit.score:.2f}）\n{hit.text}"
                for i, hit in enumerate(hits, 1)
            )

        return Tool(
            name="search_knowledge_base",
            description="在本地知识库中做语义检索，返回最相关的文档片段及来源。"
            "回答与知识库资料有关的问题前应先调用。",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "检索用的问题或关键词"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=search_knowledge_base,
        )
