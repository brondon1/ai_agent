"""命令行交互入口：`ai-agent` 或 `python -m ai_agent`。"""

from __future__ import annotations

import argparse

import anthropic
from qdrant_client import QdrantClient

from ai_agent.agent import Agent
from ai_agent.config import Config
from ai_agent.llm import AgentError
from ai_agent.rag import FastEmbedder, KnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description="ReAct + Plan-and-Solve + Reflection + RAG Agent")
    parser.add_argument(
        "--docs", nargs="*", default=[], help="导入知识库的文件或目录（.md / .txt）"
    )
    args = parser.parse_args()

    config = Config.from_env()
    knowledge = None
    if args.docs or config.qdrant_url:
        client = QdrantClient(url=config.qdrant_url) if config.qdrant_url else None
        knowledge = KnowledgeBase(
            FastEmbedder(config.embedding_model), client, config.collection, config.top_k
        )
        for path in args.docs:
            print(f"已导入 {path}：{knowledge.add_path(path)} 个片段")

    agent = Agent(config, knowledge=knowledge)
    rag = "开启" if knowledge else "关闭"
    web = "开启" if config.tavily_api_key else "关闭"
    print(f"AI Agent（模型: {config.model}，RAG: {rag}，联网搜索: {web}）。输入 /exit 退出。")
    while True:
        try:
            task = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task == "/exit":
            break
        try:
            print(f"\nAgent> {agent.run(task)}")
        except AgentError as exc:
            print(f"\n[错误] {exc}")
        except anthropic.AuthenticationError:
            print("\n[错误] 认证失败，请检查 ANTHROPIC_API_KEY")
            break
        except anthropic.RateLimitError:
            print("\n[错误] 触发限流，请稍后重试")
        except anthropic.APIStatusError as exc:
            print(f"\n[错误] API 返回 {exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError:
            print("\n[错误] 网络连接失败")


if __name__ == "__main__":
    main()
