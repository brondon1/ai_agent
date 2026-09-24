import pytest

from ai_agent.tools import Tool, ToolRegistry, default_registry


def test_calculate():
    registry = default_registry()
    assert registry.execute("calculate", {"expression": "(1 + 2) * 3"}) == ("9", False)


def test_calculate_rejects_non_arithmetic():
    result, is_error = default_registry().execute("calculate", {"expression": "__import__('os')"})
    assert is_error
    assert "ValueError" in result


def test_unknown_tool():
    assert default_registry().execute("nope", {}) == ("未知工具: nope", True)


def test_duplicate_registration():
    tool = Tool("t", "d", {"type": "object", "properties": {}}, lambda: "ok")
    registry = ToolRegistry([tool])
    with pytest.raises(ValueError):
        registry.register(tool)


def test_to_params_excludes_handler():
    params = default_registry().to_params()
    assert {p["name"] for p in params} == {"get_current_time", "calculate"}
    assert all(set(p) == {"name", "description", "input_schema"} for p in params)
