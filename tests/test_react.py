import pytest
from fakes import answer, fake_llm, response, tool_use

from ai_agent.llm import AgentError
from ai_agent.react import ReActAgent
from ai_agent.tools import default_registry


def test_plain_answer():
    llm, _ = fake_llm(answer("你好"))
    assert ReActAgent(llm, default_registry()).run("hi") == "你好"


def test_tool_call_round_trip():
    llm, client = fake_llm(
        response("tool_use", tool_use("t1", "calculate", {"expression": "2 ** 10"})),
        answer("结果是 1024"),
    )
    assert ReActAgent(llm, default_registry()).run("2 的 10 次方？") == "结果是 1024"

    messages = client.beta.messages.create.call_args.kwargs["messages"]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "1024", "is_error": False}
    ]
    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["fallbacks"] == "default"
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_refusal_raises():
    llm, _ = fake_llm(response("refusal"))
    with pytest.raises(AgentError):
        ReActAgent(llm, default_registry()).run("x")


def test_max_turns_exceeded():
    call = response("tool_use", tool_use("t", "get_current_time", {}))
    llm, _ = fake_llm(call, call)
    with pytest.raises(AgentError):
        ReActAgent(llm, default_registry(), max_turns=2).run("循环")
