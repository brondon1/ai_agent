"""基于 Claude API 的 AI Agent：ReAct + Plan-and-Solve + Reflection + RAG + Tavily 联网搜索。"""

from ai_agent.agent import Agent
from ai_agent.config import Config
from ai_agent.llm import LLM, AgentError
from ai_agent.planner import PlanAndSolve
from ai_agent.rag import FastEmbedder, KnowledgeBase
from ai_agent.react import ReActAgent
from ai_agent.reflection import Reflector
from ai_agent.tools import Tool, ToolRegistry
from ai_agent.web_search import tavily_search_tool

__all__ = [
    "LLM",
    "Agent",
    "AgentError",
    "Config",
    "FastEmbedder",
    "KnowledgeBase",
    "PlanAndSolve",
    "ReActAgent",
    "Reflector",
    "Tool",
    "ToolRegistry",
    "tavily_search_tool",
]
__version__ = "0.3.0"
