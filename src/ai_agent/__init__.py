"""基于 Claude API 的 AI Agent：ReAct + Plan-and-Solve + Reflection + RAG。"""

from ai_agent.agent import Agent
from ai_agent.config import Config
from ai_agent.llm import LLM, AgentError
from ai_agent.planner import PlanAndSolve
from ai_agent.rag import FastEmbedder, KnowledgeBase
from ai_agent.react import ReActAgent
from ai_agent.reflection import Reflector
from ai_agent.tools import Tool, ToolRegistry

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
]
__version__ = "0.2.0"
