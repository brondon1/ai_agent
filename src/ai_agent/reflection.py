"""Reflection：让模型批评自己的答案，不合格时根据反馈修改。"""

from __future__ import annotations

from ai_agent.llm import LLM
from ai_agent.react import ReActAgent

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean", "description": "答案是否已经足够好，无需修改"},
        "feedback": {"type": "string", "description": "具体的问题和修改建议；通过时可为空"},
    },
    "required": ["approved", "feedback"],
    "additionalProperties": False,
}

CRITIQUE_PROMPT = """你是严格的审稿人。检查下面的答案是否正确、完整地回答了任务，\
是否有事实错误、遗漏或没有依据的内容。只有存在实质问题时才不通过。

任务：{task}

答案：
{answer}"""

REVISE_PROMPT = """请根据审稿意见修改答案，需要时可以调用工具核实。只输出修改后的完整答案。

任务：{task}

原答案：
{answer}

审稿意见：
{feedback}"""


class Reflector:
    def __init__(self, llm: LLM, executor: ReActAgent, max_rounds: int = 1) -> None:
        self.llm = llm
        self.executor = executor
        self.max_rounds = max_rounds

    def refine(self, task: str, answer: str) -> str:
        for _ in range(self.max_rounds):
            critique = self.llm.json(
                CRITIQUE_PROMPT.format(task=task, answer=answer), CRITIQUE_SCHEMA
            )
            if critique["approved"]:
                break
            # 修改交给 ReAct 执行器，这样可以重新检索或计算来修正错误
            answer = self.executor.run(
                REVISE_PROMPT.format(task=task, answer=answer, feedback=critique["feedback"])
            )
        return answer
