import pytest
from langchain_core.tools import ToolException, tool

from browser_agent.browser_tools import page_reader


@pytest.mark.asyncio
async def test_page_reader_navigates_then_reads_content_in_one_call():
    actions = []

    @tool
    async def browser_navigate(url: str) -> str:
        """Navigate."""
        actions.append(url)
        return "Page URL: https://example.com/docs"

    @tool
    async def browser_evaluate(function: str) -> str:
        """Read page."""
        assert actions == ["https://example.com/docs"]
        return '{"title":"Graph API","url":"https://example.com/docs","text":"StateGraph"}'

    reader = page_reader([browser_navigate, browser_evaluate])
    result = await reader.ainvoke({"url": "https://example.com/docs"})
    assert "StateGraph" in str(result)
    assert "https://example.com/docs" in str(result)


@pytest.mark.asyncio
async def test_page_reader_does_not_read_stale_page_after_navigation_error():
    @tool
    async def browser_navigate(url: str) -> str:
        """Navigate."""
        raise ToolException("navigation failed")

    @tool
    async def browser_evaluate(function: str) -> str:
        """Read page."""
        pytest.fail("must not read the previous page")

    with pytest.raises(ToolException, match="navigation failed"):
        await page_reader([browser_navigate, browser_evaluate]).ainvoke({"url": "https://bad.test"})
