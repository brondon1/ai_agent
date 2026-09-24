from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_agent.agent import Agent, AgentError
from ai_agent.config import Config


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use(id_, name, tool_input):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=tool_input)


def _response(stop_reason, *content):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content))


def _agent(*responses, max_turns=20):
    client = MagicMock()
    client.beta.messages.create.side_effect = list(responses)
    return Agent(config=Config(max_turns=max_turns), client=client), client


def test_plain_answer():
    agent, _ = _agent(_response("end_turn", _text("你好")))
    assert agent.run("hi") == "你好"
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]


def test_tool_call_round_trip():
    agent, client = _agent(
        _response("tool_use", _tool_use("t1", "calculate", {"expression": "2 ** 10"})),
        _response("end_turn", _text("结果是 1024")),
    )
    assert agent.run("2 的 10 次方？") == "结果是 1024"

    tool_results = agent.messages[2]["content"]
    assert tool_results == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "1024", "is_error": False}
    ]
    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["fallbacks"] == "default"
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_refusal_rolls_back_history():
    agent, _ = _agent(
        _response("end_turn", _text("第一轮")),
        _response("refusal"),
    )
    agent.run("第一个问题")
    with pytest.raises(AgentError):
        agent.run("第二个问题")
    assert len(agent.messages) == 2


def test_max_turns_exceeded():
    call = _response("tool_use", _tool_use("t", "get_current_time", {}))
    agent, _ = _agent(call, call, max_turns=2)
    with pytest.raises(AgentError):
        agent.run("循环")
    assert agent.messages == []
