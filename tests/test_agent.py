from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from browser_agent.graph import build_graph
from browser_agent.skills import load_skills


class ScriptedModel:
    """Replace only the paid model; the graph and tools are real."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []
        self.tool_inputs = []

    def bind_tools(self, tools):
        return SimpleNamespace(ainvoke=self.invoke_with_tools)

    async def invoke_with_tools(self, messages):
        self.tool_inputs.append(messages)
        return await self.ainvoke(messages)

    async def ainvoke(self, messages):
        self.inputs.append(messages)
        return next(self.responses)


def call(name, args=None, id="call1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": id}])


@pytest.mark.asyncio
async def test_graph_loads_strategy_and_uses_tool_observation():
    @tool
    def browser_navigate(url: str) -> str:
        """Open a page."""
        return f"Page title: Example at {url}"

    model = ScriptedModel(
        [
            call("browser_navigate", {"url": "https://example.com"}),
            AIMessage(content="页面标题是 Example"),
        ]
    )
    graph = build_graph(model, [browser_navigate], load_skills(Path("skills")))
    result = await graph.ainvoke({"task": "搜索 example", "max_steps": 3})
    assert result["status"] == "completed"
    assert result["steps"] == 2
    assert result["answer"] == "页面标题是 Example"
    assert "search" in result["selected_skills"]
    assert "搜索关键词" in model.inputs[0][0].content
    observations = [m for m in model.inputs[1] if isinstance(m, ToolMessage)]
    assert observations[0].tool_call_id == "call1"
    assert "Page title: Example" in observations[0].content


@pytest.mark.asyncio
async def test_budget_stops_before_extra_tool_execution():
    visited = []

    @tool
    def browser_click() -> str:
        """Click a button."""
        visited.append("click")
        return "clicked"

    model = ScriptedModel([call("browser_click"), AIMessage(content="已点击，尚未确认完成")])
    result = await build_graph(model, [browser_click], []).ainvoke(
        {"task": "点击按钮", "max_steps": 2}
    )
    assert result["status"] == "stopped"
    assert visited == ["click"]
    assert result["answer"]


@pytest.mark.asyncio
async def test_invalid_tool_and_arguments_are_observations():
    @tool
    def browser_navigate(url: str) -> str:
        """Open URL."""
        return url

    model = ScriptedModel(
        [call("missing"), call("browser_navigate", id="call2"), AIMessage(content="无法继续")]
    )
    result = await build_graph(model, [browser_navigate], []).ainvoke(
        {"task": "打开页面", "max_steps": 4}
    )
    errors = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(errors) == 2
    assert all(m.status == "error" for m in errors)


@pytest.mark.asyncio
async def test_browser_calls_execute_in_order():
    history = []

    @tool
    async def browser_action(value: int) -> str:
        """Record a browser action."""
        import asyncio

        await asyncio.sleep(0.01 if value == 1 else 0)
        history.append(value)
        return str(history)

    response = AIMessage(
        content="",
        tool_calls=[
            {"name": "browser_action", "args": {"value": 1}, "id": "a"},
            {"name": "browser_action", "args": {"value": 2}, "id": "b"},
        ],
    )
    result = await build_graph(
        ScriptedModel([response, AIMessage(content="完成")]), [browser_action], []
    ).ainvoke({"task": "操作", "max_steps": 3})
    assert result["status"] == "completed"
    assert history == [1, 2]


@pytest.mark.asyncio
async def test_budget_reserves_last_call_for_evidence_summary():
    @tool
    def browser_navigate(url: str) -> str:
        """Open a documentation page."""
        return "Graph API https://docs.example/graph — StateGraph and edges"

    model = ScriptedModel(
        [
            call("browser_navigate", {"url": "https://docs.example/graph"}),
            AIMessage(
                content="已找到 [Graph API](https://docs.example/graph)，工具调用页面尚未核实。"
            ),
        ]
    )
    result = await build_graph(model, [browser_navigate], []).ainvoke(
        {"task": "查找状态图和工具文档", "max_steps": 2}
    )
    assert result["status"] == "stopped"
    assert result["steps"] == 2
    assert "https://docs.example/graph" in result["answer"]
    assert "尚未核实" in result["answer"]
    assert len(model.tool_inputs) == 1
    assert "不再调用工具" in model.inputs[-1][-1].content
    assert any(isinstance(m, ToolMessage) and "StateGraph" in m.content for m in model.inputs[-1])


@pytest.mark.asyncio
async def test_identical_actions_and_observations_end_early():
    @tool
    def browser_navigate(url: str) -> str:
        """Navigate."""
        return "Access denied"

    model = ScriptedModel(
        [call("browser_navigate", {"url": "https://example.com"}, id=str(i)) for i in range(3)]
        + [AIMessage(content="页面拒绝访问，未获得资料。")]
    )
    result = await build_graph(model, [browser_navigate], []).ainvoke(
        {"task": "搜索文档", "max_steps": 15}
    )
    assert result["status"] == "stopped"
    assert result["steps"] == 4
    assert "重复" in result["stop_reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocks", [False, True])
async def test_large_observations_are_bounded_without_changing_execution_record(blocks):
    content = "Document title\n" + "navigation " * 3000 + "\nhttps://example.com/source"

    original = [{"type": "text", "text": content, "id": "mcp-block"}] if blocks else content

    @tool
    def browser_read() -> str | list:
        """Read page."""
        return original

    model = ScriptedModel([call("browser_read"), AIMessage(content="完成")])
    result = await build_graph(model, [browser_read], []).ainvoke(
        {"task": "读取页面", "max_steps": 3}
    )
    sent = next(m for m in model.inputs[-1] if isinstance(m, ToolMessage))
    stored = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert len(sent.content) < 13000
    assert "截断" in sent.content
    assert "https://example.com/source" in sent.content
    assert stored.content == original


@pytest.mark.asyncio
async def test_token_usage_includes_final_summary():
    response = AIMessage(
        content="完成",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
        },
    )
    result = await build_graph(ScriptedModel([response]), [], []).ainvoke(
        {"task": "描述任务", "max_steps": 1}
    )
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 20


@pytest.mark.asyncio
async def test_repeated_action_with_changing_content_is_not_stalled():
    observations = iter(["loading", "partially loaded", "complete"])

    @tool
    async def browser_read() -> str:
        """Read changing content."""
        return next(observations)

    model = ScriptedModel(
        [call("browser_read", id=str(i)) for i in range(3)] + [AIMessage(content="已读到完整内容")]
    )
    result = await build_graph(model, [browser_read], []).ainvoke(
        {"task": "读取动态内容", "max_steps": 6}
    )
    assert result["status"] == "completed"
    assert result["steps"] == 4
