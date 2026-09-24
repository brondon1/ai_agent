"""工具定义与注册表。"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]

    def to_param(self) -> dict[str, Any]:
        """转换为 Messages API 的 tools 参数格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具已存在: {tool.name}")
        self._tools[tool.name] = tool

    def to_params(self) -> list[dict[str, Any]]:
        return [tool.to_param() for tool in self._tools.values()]

    def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """执行工具，返回 (结果文本, 是否出错)。异常会转成错误结果交还给模型。"""
        tool = self._tools.get(name)
        if tool is None:
            return f"未知工具: {name}", True
        try:
            return tool.handler(**tool_input), False
        except Exception as exc:  # noqa: BLE001 - 错误信息回传给模型处理
            return f"{type(exc).__name__}: {exc}", True


# ---- 内置示例工具 ----


def _current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


_OPERATORS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("只支持数字和 + - * / // % ** 运算")


def _calculate(expression: str) -> str:
    """安全地计算算术表达式（不使用 eval）。"""
    return str(_eval_node(ast.parse(expression, mode="eval").body))


current_time_tool = Tool(
    name="get_current_time",
    description="获取当前的 UTC 时间（ISO 8601 格式）。",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    handler=_current_time,
)

calculator_tool = Tool(
    name="calculate",
    description="计算一个算术表达式，支持 + - * / // % ** 和括号。",
    input_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "例如 (1 + 2) * 3"},
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    handler=_calculate,
)


def default_registry() -> ToolRegistry:
    return ToolRegistry([current_time_tool, calculator_tool])
