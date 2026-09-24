"""运行配置，从环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # 模型
    model: str = "claude-opus-5"
    max_tokens: int = 16000
    effort: str = "high"  # low | medium | high | xhigh | max
    # ReAct：单次执行中最多的模型调用轮数，防止工具调用死循环
    max_turns: int = 20
    # Plan-and-Solve
    use_planning: bool = True
    max_plan_steps: int = 5
    # Reflection：最多自我批评 / 修改几轮，0 表示关闭
    max_reflections: int = 1
    # RAG：qdrant_url 为空时使用内存模式（进程退出即丢失）
    qdrant_url: str | None = None
    collection: str = "knowledge"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    top_k: int = 4
    # Tavily 联网搜索：设置了 key 才会启用 web_search 工具
    tavily_api_key: str | None = field(default=None, repr=False)
    tavily_max_results: int = 5

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            model=os.getenv("AI_AGENT_MODEL", cls.model),
            effort=os.getenv("AI_AGENT_EFFORT", cls.effort),
            max_turns=int(os.getenv("AI_AGENT_MAX_TURNS", cls.max_turns)),
            use_planning=os.getenv("AI_AGENT_USE_PLANNING", "1") not in ("0", "false", "False"),
            max_plan_steps=int(os.getenv("AI_AGENT_MAX_PLAN_STEPS", cls.max_plan_steps)),
            max_reflections=int(os.getenv("AI_AGENT_MAX_REFLECTIONS", cls.max_reflections)),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            collection=os.getenv("AI_AGENT_COLLECTION", cls.collection),
            embedding_model=os.getenv("AI_AGENT_EMBEDDING_MODEL", cls.embedding_model),
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            tavily_max_results=int(
                os.getenv("AI_AGENT_TAVILY_MAX_RESULTS", cls.tavily_max_results)
            ),
        )
