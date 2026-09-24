"""把 ReAct、Plan-and-Solve、Reflection 和 RAG 组装在一起的 Agent。

流程：
    任务 ──► Plan-and-Solve 拆分步骤
              └─ 每一步交给 ReAct 执行（可调用 search_knowledge_base、web_search 等工具）
         ──► 汇总得到初稿
         ──► Reflection 自我批评，不通过则由 ReAct 修改
         ──► 最终答案
"""

from __future__ import annotations

import anthropic

from ai_agent.config import Config
from ai_agent.llm import LLM
from ai_agent.planner import PlanAndSolve
from ai_agent.rag import KnowledgeBase
from ai_agent.react import ReActAgent
from ai_agent.reflection import Reflector
from ai_agent.tools import ToolRegistry, default_registry
from ai_agent.web_search import tavily_search_tool


class Agent:
    def __init__(
        self,
        config: Config | None = None,
        tools: ToolRegistry | None = None,
        knowledge: KnowledgeBase | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.config = config or Config.from_env()
        self.tools = tools if tools is not None else default_registry()
        if knowledge is not None:
            self.tools.register(knowledge.as_tool())
        if self.config.tavily_api_key:
            self.tools.register(
                tavily_search_tool(self.config.tavily_api_key, self.config.tavily_max_results)
            )

        self.llm = LLM(self.config, client)
        self.react = ReActAgent(self.llm, self.tools, self.config.max_turns)
        self.planner = PlanAndSolve(self.llm, self.react, self.config.max_plan_steps)
        self.reflector = Reflector(self.llm, self.react, self.config.max_reflections)

    def run(self, task: str) -> str:
        if self.config.use_planning:
            answer = self.planner.run(task)
        else:
            answer = self.react.run(task)
        return self.reflector.refine(task, answer)
