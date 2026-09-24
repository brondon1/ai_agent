# ai_agent

基于 [Claude API](https://docs.claude.com) 的 Python AI Agent 基础项目：一个带工具调用循环的最小 Agent，外加命令行交互入口。

## 目录结构

```
ai_agent/
├── pyproject.toml        # 项目元数据、依赖、ruff / pytest 配置
├── .env.example          # 环境变量示例
├── src/ai_agent/
│   ├── agent.py          # Agent 主循环：调用模型 → 执行工具 → 回传结果
│   ├── tools.py          # Tool / ToolRegistry 及内置示例工具
│   ├── config.py         # 配置（模型、effort、最大轮数、系统提示词）
│   ├── cli.py            # 命令行交互
│   └── __main__.py       # 支持 python -m ai_agent
└── tests/                # 单元测试（mock 掉 API，不需要密钥）
```

## 快速开始

需要 Python 3.10+。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export ANTHROPIC_API_KEY=your-api-key   # 或者用 `ant auth login` 登录
ai-agent                                # 也可以 python -m ai_agent
```

## 在代码里使用

```python
from ai_agent import Agent

agent = Agent()
print(agent.run("现在几点？再帮我算一下 (3 + 4) * 12"))
```

## 添加自定义工具

```python
from ai_agent import Agent, Tool
from ai_agent.tools import default_registry


def get_weather(city: str) -> str:
    return f"{city}：晴，25°C"


registry = default_registry()
registry.register(
    Tool(
        name="get_weather",
        description="查询城市的天气。",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        handler=get_weather,
    )
)

agent = Agent(tools=registry)
```

工具抛出的异常会作为 `is_error` 的结果交还给模型，由模型决定如何处理。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | API 密钥 |
| `AI_AGENT_MODEL` | `claude-opus-5` | 模型 ID |
| `AI_AGENT_EFFORT` | `high` | 推理强度：`low` / `medium` / `high` / `xhigh` / `max` |
| `AI_AGENT_MAX_TURNS` | `20` | 单次请求最多的模型调用轮数 |

默认开启了自适应思考（adaptive thinking）和服务端拒答回退（`fallbacks: "default"`）：模型拒答时，API 会自动换用推荐模型重试。

## 开发

```bash
pytest          # 运行测试
ruff check .    # 代码检查
ruff format .   # 格式化
```
