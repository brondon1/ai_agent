from fakes import answer, fake_llm, json_answer

from ai_agent.planner import PlanAndSolve
from ai_agent.react import ReActAgent
from ai_agent.tools import default_registry


def _planner(*responses, max_steps=5):
    llm, client = fake_llm(*responses)
    return PlanAndSolve(llm, ReActAgent(llm, default_registry()), max_steps), client


def test_multi_step_plan_executes_and_synthesizes():
    planner, client = _planner(
        json_answer({"steps": ["查资料", "写总结"]}),
        answer("资料内容"),
        answer("总结草稿"),
        answer("最终答案"),
    )
    assert planner.run("任务") == "最终答案"

    calls = client.beta.messages.create.call_args_list
    assert calls[0].kwargs["output_config"]["format"]["type"] == "json_schema"
    second_step_prompt = calls[2].kwargs["messages"][0]["content"]
    assert "资料内容" in second_step_prompt and "第 2/2 步" in second_step_prompt


def test_single_step_plan_skips_synthesis():
    planner, client = _planner(json_answer({"steps": ["直接回答"]}), answer("答案"))
    assert planner.run("1+1？") == "答案"
    assert client.beta.messages.create.call_count == 2


def test_plan_is_capped():
    planner, _ = _planner(json_answer({"steps": ["a", "b", "c"]}), max_steps=2)
    assert planner.plan("任务") == ["a", "b"]
