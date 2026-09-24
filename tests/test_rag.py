import pytest
from fakes import HashEmbedder

from ai_agent.rag import KnowledgeBase, chunk_text


def test_chunk_text_merges_paragraphs_and_splits_long_ones():
    assert chunk_text("甲\n\n乙", size=10, overlap=2) == ["甲\n\n乙"]
    chunks = chunk_text("x" * 25, size=10, overlap=2)
    assert all(len(c) <= 10 for c in chunks)
    assert "".join(c[2:] if i else c for i, c in enumerate(chunks)) == "x" * 25


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", size=5, overlap=5)


def test_search_returns_most_relevant_chunk(tmp_path):
    kb = KnowledgeBase(HashEmbedder())
    kb.add_texts(["Qdrant 是一个向量数据库，用于相似度检索。"], source="qdrant.md")
    (tmp_path / "cat.txt").write_text("猫是一种常见的宠物，喜欢睡觉。", encoding="utf-8")
    assert kb.add_path(tmp_path) == 1

    hits = kb.search("什么是向量数据库", k=1)
    assert hits[0].source == "qdrant.md"


def test_tool_output():
    kb = KnowledgeBase(HashEmbedder())
    tool = kb.as_tool()
    assert tool.handler(query="任何问题") == "知识库中没有找到相关内容。"
    kb.add_texts(["ReAct 交替进行推理和行动。"], source="react.md")
    assert "来源：react.md" in tool.handler(query="ReAct 是什么")
