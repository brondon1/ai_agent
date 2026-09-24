# ai_agent

基于 [Claude API](https://docs.claude.com) 的最简 AI Agent 架构，把四种常见模式组合在一起：

| 模式 | 作用 | 代码 |
| --- | --- | --- |
| **ReAct** | 推理与行动交替：思考 → 调用工具 → 观察结果 → 继续思考 | `react.py` |
| **Plan-and-Solve** | 先把任务拆成步骤，逐步交给 ReAct 执行，最后汇总 | `planner.py` |
| **Reflection** | 自我批评答案，不通过时根据反馈修改 | `reflection.py` |
| **RAG + Qdrant** | 文档切块向量化存入 Qdrant，作为 `search_knowledge_base` 工具供 ReAct 调用 | `rag/` |

## 执行流程

```
用户任务
  │
  ▼
Plan-and-Solve ── 生成计划（结构化输出：steps 列表）
  │
  ├─ 第 1 步 ──► ReAct ──► 工具：search_knowledge_base（Qdrant） / calculate / ...
  ├─ 第 2 步 ──► ReAct ──► ...（带上前面步骤的结果）
  │
  ▼
汇总成初稿（单步计划时跳过）
  │
  ▼
Reflection ── 审稿（结构化输出：approved + feedback）
  │   └─ 不通过 → 交给 ReAct 修改（可重新检索）→ 再审，最多 N 轮
  ▼
最终答案
```

## 目录结构

```
src/ai_agent/
├── agent.py          # Agent：把下面各模块组装成完整流程
├── llm.py            # 对 Claude API 的薄封装（自适应思考、结构化输出、拒答回退）
├── react.py          # ReAct 执行器
├── planner.py        # Plan-and-Solve
├── reflection.py     # Reflection
├── rag/
│   ├── embeddings.py # 向量模型（默认 FastEmbed 本地运行 bge-small-zh）
│   └── knowledge.py  # 切块、写入 Qdrant、检索、包装成工具
├── tools.py          # 工具注册表 + 示例工具（当前时间、计算器）
├── config.py         # 配置
└── cli.py            # 命令行入口
tests/                # 单元测试（模拟 API 和向量模型，不需要密钥和网络）
```

## 快速开始

需要 Python 3.10+。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=your-api-key

ai-agent                      # 不带知识库
ai-agent --docs ./docs        # 导入 ./docs 下的 .md / .txt 文件，开启 RAG
```

首次开启 RAG 时，FastEmbed 会从 Hugging Face 下载向量模型（约 100MB）。

### 使用 Qdrant 服务（持久化）

默认 Qdrant 运行在内存模式，进程退出后数据丢失。要持久化，先启动 Qdrant：

```bash
docker run -p 6333:6333 qdrant/qdrant
export QDRANT_URL=http://localhost:6333
ai-agent --docs ./docs        # 导入一次即可，之后直接 ai-agent 也会连上已有的知识库
```

## 在代码里使用

```python
from ai_agent import Agent, Config, FastEmbedder, KnowledgeBase

kb = KnowledgeBase(FastEmbedder())
kb.add_texts(["Qdrant 是一个开源的向量数据库。"], source="notes")
kb.add_path("./docs")

agent = Agent(Config(max_reflections=2), knowledge=kb)
print(agent.run("Qdrant 是什么？适合用在什么场景？"))
```

各模块也可以单独使用，比如只要 ReAct：

```python
from ai_agent import LLM, Config, ReActAgent
from ai_agent.tools import default_registry

react = ReActAgent(LLM(Config()), default_registry())
print(react.run("(3 + 4) * 12 等于多少？"))
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
| `AI_AGENT_MAX_TURNS` | `20` | ReAct 单次执行最多的模型调用轮数 |
| `AI_AGENT_USE_PLANNING` | `1` | 设为 `0` 关闭 Plan-and-Solve，直接用 ReAct |
| `AI_AGENT_MAX_PLAN_STEPS` | `5` | 计划最多几步 |
| `AI_AGENT_MAX_REFLECTIONS` | `1` | 最多反思几轮，`0` 关闭 |
| `QDRANT_URL` | 空（内存模式） | Qdrant 服务地址 |
| `AI_AGENT_COLLECTION` | `knowledge` | Qdrant 集合名 |
| `AI_AGENT_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | FastEmbed 向量模型 |

每个问题的模型调用次数大约是：1 次规划 + 每步若干次 ReAct + 1 次汇总 + 每轮反思 1～2 次。简单问题可以关掉规划或反思来省钱、提速。

默认开启了自适应思考（adaptive thinking）和服务端拒答回退（`fallbacks: "default"`）：模型拒答时，API 会自动换用推荐模型重试。

## 开发

```bash
pytest          # 运行测试
ruff check .    # 代码检查
ruff format .   # 格式化
```
