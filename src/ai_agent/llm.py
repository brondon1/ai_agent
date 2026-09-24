"""对 Claude Messages API 的薄封装，供各个模块共用。"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from ai_agent.config import Config

# 服务端拒答回退：模型因安全分类器拒答时，由 API 按拒答类别自动换用推荐模型重试
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AgentError(RuntimeError):
    pass


class LLM:
    def __init__(self, config: Config, client: anthropic.Anthropic | None = None) -> None:
        self.config = config
        # 默认从环境变量 / `ant auth login` 配置中读取凭据
        self.client = client or anthropic.Anthropic()

    def create(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> Any:
        """调用一次模型，处理拒答和截断，返回原始 response。"""
        output_config: dict[str, Any] = {"effort": self.config.effort}
        if output_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}

        kwargs: dict[str, Any] = {}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self.client.beta.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            thinking={"type": "adaptive"},
            output_config=output_config,
            messages=messages,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise AgentError("模型拒绝了该请求")
        if response.stop_reason == "max_tokens":
            raise AgentError("输出达到 max_tokens 上限，请调大 max_tokens")
        return response

    def text(self, prompt: str, *, system: str | None = None) -> str:
        response = self.create([{"role": "user", "content": prompt}], system=system)
        return text_of(response.content)

    def json(
        self, prompt: str, schema: dict[str, Any], *, system: str | None = None
    ) -> dict[str, Any]:
        """结构化输出：返回符合 schema 的 JSON 对象。"""
        response = self.create(
            [{"role": "user", "content": prompt}], system=system, output_schema=schema
        )
        return json.loads(text_of(response.content))


def text_of(content: list[Any]) -> str:
    return "".join(block.text for block in content if block.type == "text")
