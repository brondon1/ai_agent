from fakes import answer, fake_llm, json_answer

from ai_agent.react import ReActAgent
from ai_agent.reflection import Reflector
from ai_agent.tools import default_registry


def _reflector(*responses, max_rounds=1):
    llm, client = fake_llm(*responses)
    return Reflector(llm, ReActAgent(llm, default_registry()), max_rounds), client


def test_approved_answer_is_kept():
    reflector, client = _reflector(json_answer({"approved": True, "feedback": ""}))
    assert reflector.refine("任务", "初稿") == "初稿"
    assert client.beta.messages.create.call_count == 1


def test_rejected_answer_is_revised():
    reflector, client = _reflector(
        json_answer({"approved": False, "feedback": "漏了单位"}),
        answer("修改后的答案"),
    )
    assert reflector.refine("任务", "初稿") == "修改后的答案"
    revise_prompt = client.beta.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "漏了单位" in revise_prompt


def test_disabled_reflection_makes_no_calls():
    reflector, client = _reflector(max_rounds=0)
    assert reflector.refine("任务", "初稿") == "初稿"
    client.beta.messages.create.assert_not_called()
