"""运行配置，从环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人的 AI 助手。需要时调用可用的工具，并用中文回答。"


@dataclass(frozen=True)
class Config:
    model: str = "claude-opus-5"
    max_tokens: int = 16000
    effort: str = "high"  # low | medium | high | xhigh | max
    max_turns: int = 20  # 单次 run 中最多的模型调用轮数，防止工具调用死循环
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            model=os.getenv("AI_AGENT_MODEL", cls.model),
            effort=os.getenv("AI_AGENT_EFFORT", cls.effort),
            max_turns=int(os.getenv("AI_AGENT_MAX_TURNS", cls.max_turns)),
        )
