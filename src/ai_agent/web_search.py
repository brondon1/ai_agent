"""联网搜索：基于 Tavily 的 web_search 工具。"""

from __future__ import annotations

from typing import Any

from ai_agent.tools import Tool

MAX_CONTENT_CHARS = 800


def tavily_search_tool(
    api_key: str | None = None,
    max_results: int = 5,
    search_depth: str = "basic",
    client: Any = None,
) -> Tool:
    """创建 web_search 工具。api_key 为空时由 Tavily SDK 读取环境变量 TAVILY_API_KEY。"""
    if client is None:
        from tavily import TavilyClient  # 延迟导入

        client = TavilyClient(api_key=api_key)

    def web_search(query: str, topic: str = "general") -> str:
        response = client.search(
            query,
            topic=topic,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
        )
        parts = []
        if response.get("answer"):
            parts.append(f"摘要：{response['answer']}")
        for i, result in enumerate(response.get("results", []), 1):
            content = (result.get("content") or "")[:MAX_CONTENT_CHARS]
            parts.append(f"[{i}] {result.get('title', '')}\n{result.get('url', '')}\n{content}")
        return "\n\n".join(parts) or "没有找到相关的搜索结果。"

    return Tool(
        name="web_search",
        description="使用 Tavily 联网搜索，返回摘要和带链接的网页片段。"
        "用于知识库之外、需要最新信息或公开资料的问题；引用结果时注明链接。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词或问题"},
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                    "description": "搜索类别：通用、新闻或财经，默认 general",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=web_search,
    )
