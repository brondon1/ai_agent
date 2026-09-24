"""命令行交互入口：`ai-agent` 或 `python -m ai_agent`。"""

from __future__ import annotations

import anthropic

from ai_agent.agent import Agent, AgentError


def main() -> None:
    agent = Agent()
    print(f"AI Agent（模型: {agent.config.model}）。输入 /reset 清空对话，/exit 退出。")
    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/reset":
            agent.reset()
            print("对话已清空。")
            continue
        try:
            print(f"\nAgent> {agent.run(user_input)}")
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
