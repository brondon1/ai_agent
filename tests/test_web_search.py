from unittest.mock import MagicMock

from fakes import answer, fake_client, json_answer, response, tool_use

from ai_agent.agent import Agent
from ai_agent.config import Config
from ai_agent.tools import ToolRegistry
from ai_agent.web_search import tavily_search_tool


def _tavily(result):
    client = MagicMock()
    client.search.return_value = result
    return client


def test_formats_answer_and_results():
    client = _tavily(
        {
            "answer": "Qdrant 是向量数据库。",
            "results": [{"title": "Qdrant", "url": "https://qdrant.tech", "content": "x" * 2000}],
        }
    )
    tool = tavily_search_tool(max_results=3, client=client)
    output = tool.handler(query="Qdrant 是什么", topic="news")

    assert output.startswith("摘要：Qdrant 是向量数据库。")
    assert "[1] Qdrant\nhttps://qdrant.tech" in output
    assert len(output) < 1000  # 网页内容被截断
    client.search.assert_called_once_with(
        "Qdrant 是什么", topic="news", max_results=3, search_depth="basic", include_answer=True
    )


def test_empty_results():
    tool = tavily_search_tool(client=_tavily({"results": []}))
    assert tool.handler(query="没有结果") == "没有找到相关的搜索结果。"


def test_errors_are_returned_to_model():
    client = MagicMock()
    client.search.side_effect = RuntimeError("Unauthorized")
    registry = ToolRegistry([tavily_search_tool(client=client)])
    assert registry.execute("web_search", {"query": "x"}) == ("RuntimeError: Unauthorized", True)


def test_agent_registers_web_search_only_with_key():
    without = Agent(Config(), client=fake_client())
    assert "web_search" not in {p["name"] for p in without.tools.to_params()}

    with_key = Agent(Config(tavily_api_key="tvly-test"), client=fake_client())
    assert "web_search" in {p["name"] for p in with_key.tools.to_params()}
    assert "tvly-test" not in repr(with_key.config)


def test_react_uses_web_search(monkeypatch):
    tavily = _tavily(
        {"results": [{"title": "新闻", "url": "https://e.com", "content": "今天的新闻"}]}
    )
    monkeypatch.setattr("tavily.TavilyClient", lambda api_key=None: tavily)
    client = fake_client(
        response("tool_use", tool_use("t1", "web_search", {"query": "今天的新闻"})),
        answer("根据 https://e.com：今天的新闻"),
        json_answer({"approved": True, "feedback": ""}),
    )
    agent = Agent(Config(use_planning=False, tavily_api_key="tvly-test"), client=client)

    assert agent.run("今天有什么新闻？") == "根据 https://e.com：今天的新闻"
    observation = client.beta.messages.create.call_args_list[1].kwargs["messages"][2]
    assert "https://e.com" in observation["content"][0]["content"]
