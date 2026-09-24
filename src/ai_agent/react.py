"""ReAct：推理（Reason）与行动（Act）交替进行。

模型先思考（adaptive thinking），再决定调用哪个工具；工具结果作为观察（Observation）
回传给模型，循环直到模型不再调用工具、给出答案。
"""

from __future__ import annotations

from typing import Any

from ai_agent.llm import LLM, AgentError, text_of
from ai_agent.tools import ToolRegistry

SYSTEM_PROMPT = """你是一个按 ReAct 方式工作的助手：先思考需要什么信息，再调用工具获取，\
根据工具返回的结果继续思考，直到能给出答案。
- 涉及知识库里的资料时，先用 search_knowledge_base 检索，并以检索结果为依据
- 知识库没有、或需要最新信息时，用 web_search 联网搜索，并注明来源链接
- 需要计算时使用 calculate，不要心算
- 用中文回答"""


class ReActAgent:
    def __init__(self, llm: LLM, tools: ToolRegistry, max_turns: int = 20) -> None:
        self.llm = llm
        self.tools = tools
        self.max_turns = max_turns

    def run(self, task: str) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

        for _ in range(self.max_turns):
            response = self.llm.create(messages, system=SYSTEM_PROMPT, tools=self.tools.to_params())
            # 保留完整的 content（含 thinking / tool_use 块），而不只是文本
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason != "tool_use":
                return text_of(response.content)

            observations = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result, is_error = self.tools.execute(block.name, block.input)
                observations.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        "is_error": is_error,
                    }
                )
            # 所有工具结果放在同一条 user 消息里返回
            messages.append({"role": "user", "content": observations})

        raise AgentError(f"超过最大轮数 {self.max_turns}，仍未得到最终回答")
