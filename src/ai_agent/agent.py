"""Agent 主循环：调用 Claude，执行工具，直到模型给出最终回答。"""

from __future__ import annotations

from typing import Any

import anthropic

from ai_agent.config import Config
from ai_agent.tools import ToolRegistry, default_registry

# 服务端拒答回退：模型因安全分类器拒答时，由 API 按拒答类别自动换用推荐模型重试
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AgentError(RuntimeError):
    pass


class Agent:
    def __init__(
        self,
        config: Config | None = None,
        tools: ToolRegistry | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.config = config or Config.from_env()
        self.tools = tools if tools is not None else default_registry()
        # 默认从环境变量 / `ant auth login` 配置中读取凭据
        self.client = client or anthropic.Anthropic()
        self.messages: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.messages = []

    def run(self, user_input: str) -> str:
        """发送一条用户消息，跑完工具调用循环，返回最终文本。

        成功时对话历史会保留；失败时回滚本次 run 追加的消息，保证历史始终合法。
        """
        start = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            return self._loop()
        except BaseException:
            del self.messages[start:]
            raise

    def _loop(self) -> str:
        for _ in range(self.config.max_turns):
            response = self.client.beta.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                thinking={"type": "adaptive"},
                output_config={"effort": self.config.effort},
                tools=self.tools.to_params(),
                messages=self.messages,
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
            # 保留完整的 content（含 thinking / tool_use 块），而不只是文本
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                raise AgentError("模型拒绝了该请求")
            if response.stop_reason == "max_tokens":
                raise AgentError("输出达到 max_tokens 上限，请调大 max_tokens")
            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason != "tool_use":
                return _text_of(response.content)

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result, is_error = self.tools.execute(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        "is_error": is_error,
                    }
                )
            # 所有工具结果放在同一条 user 消息里返回
            self.messages.append({"role": "user", "content": tool_results})

        raise AgentError(f"超过最大轮数 {self.config.max_turns}，仍未得到最终回答")


def _text_of(content: list[Any]) -> str:
    return "".join(block.text for block in content if block.type == "text")
