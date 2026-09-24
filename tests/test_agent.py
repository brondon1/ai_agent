from fakes import HashEmbedder, answer, fake_client, json_answer, response, tool_use

from ai_agent.agent import Agent
from ai_agent.config import Config
from ai_agent.rag import KnowledgeBase


def test_full_pipeline_with_rag():
    kb = KnowledgeBase(HashEmbedder())
    kb.add_texts(["本项目的默认模型是 claude-opus-5。"], source="readme.md")
    client = fake_client(
        # Plan-and-Solve：单步计划
        json_answer({"steps": ["检索默认模型"]}),
        # ReAct：先检索，再回答
        response("tool_use", tool_use("t1", "search_knowledge_base", {"query": "默认模型"})),
        answer("初稿"),
        # Reflection：不通过 → 修改
        json_answer({"approved": False, "feedback": "要注明来源"}),
        answer("默认模型是 claude-opus-5（来源：readme.md）"),
    )
    agent = Agent(Config(), knowledge=kb, client=client)

    assert agent.run("默认模型是什么？") == "默认模型是 claude-opus-5（来源：readme.md）"
    observation = client.beta.messages.create.call_args_list[2].kwargs["messages"][2]
    assert "readme.md" in observation["content"][0]["content"]


def test_react_only_without_planning_or_reflection():
    client = fake_client(answer("直接回答"))
    agent = Agent(Config(use_planning=False, max_reflections=0), client=client)
    assert agent.run("你好") == "直接回答"
    assert client.beta.messages.create.call_count == 1
