"""Plan-and-Solve：先把任务拆成步骤，再逐步执行，最后汇总。"""

from __future__ import annotations

from ai_agent.llm import LLM
from ai_agent.react import ReActAgent

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "按顺序执行的步骤，每步是一句可独立执行的指令",
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}

PLAN_PROMPT = """请为下面的任务制定一个简洁的执行计划。
- 简单任务只需要 1 步，不要为了拆分而拆分
- 最多 {max_steps} 步，每一步都要具体、可执行
- 可用工具：{tools}

任务：{task}"""

SOLVE_PROMPT = """总任务：{task}

已完成的步骤及结果：
{history}

现在执行第 {index}/{total} 步：{step}"""

SYNTHESIZE_PROMPT = """根据各步骤的执行结果，给出总任务的最终回答。直接回答，不要复述过程。

总任务：{task}

各步骤结果：
{history}"""


class PlanAndSolve:
    def __init__(self, llm: LLM, executor: ReActAgent, max_steps: int = 5) -> None:
        self.llm = llm
        self.executor = executor
        self.max_steps = max_steps

    def plan(self, task: str) -> list[str]:
        tools = "、".join(p["name"] for p in self.executor.tools.to_params()) or "无"
        prompt = PLAN_PROMPT.format(max_steps=self.max_steps, tools=tools, task=task)
        steps = [s.strip() for s in self.llm.json(prompt, PLAN_SCHEMA)["steps"] if s.strip()]
        return steps[: self.max_steps] or [task]

    def run(self, task: str) -> str:
        steps = self.plan(task)
        if len(steps) == 1:
            # 单步计划没有必要再汇总，直接交给执行器处理原任务
            return self.executor.run(task)

        results: list[tuple[str, str]] = []
        for index, step in enumerate(steps, 1):
            prompt = SOLVE_PROMPT.format(
                task=task,
                history=_format(results) or "（无）",
                index=index,
                total=len(steps),
                step=step,
            )
            results.append((step, self.executor.run(prompt)))

        return self.llm.text(SYNTHESIZE_PROMPT.format(task=task, history=_format(results)))


def _format(results: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"{i}. {step}\n结果：{result}" for i, (step, result) in enumerate(results, 1)
    )
