import sys
from pathlib import Path

import pytest
from mcp import StdioServerParameters

from browser_agent.runtime import Settings, connect_tools, run_task


def test_settings_reject_missing_key_and_bad_url():
    with pytest.raises(ValueError, match="API Key"):
        Settings(api_key="", base_url="https://example.com", model="model")
    with pytest.raises(ValueError, match="URL"):
        Settings(api_key="test", base_url="example.com", model="model")


async def test_mcp_session_preserves_state_across_calls():
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("mcp_server.py"))],
    )
    async with connect_tools(server) as tools:
        counter = next(t for t in tools if t.name == "increment")
        first = await counter.ainvoke({})
        second = await counter.ainvoke({})
        assert "1" in str(first)
        assert "2" in str(second)


async def test_runtime_converts_connection_error_to_failed_task(monkeypatch):
    from contextlib import asynccontextmanager

    import browser_agent.runtime as runtime

    @asynccontextmanager
    async def unavailable(server):
        raise OSError("MCP unavailable")
        yield

    monkeypatch.setattr(runtime, "browser_server", lambda *args: None)
    monkeypatch.setattr(runtime, "connect_tools", unavailable)
    events = []
    result = await run_task(
        "打开首页", Settings("test", "https://example.com", "model"), on_update=events.append
    )
    assert result["status"] == "failed"
    assert "MCP unavailable" in result["answer"]
    assert events[-1]["status"] == "failed"
