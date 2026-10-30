"""Local MCP fixture: no browser or online model required."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("test-session")
count = 0


@server.tool()
def increment() -> str:
    """Increment a session-local counter."""
    global count
    count += 1
    return str(count)


if __name__ == "__main__":
    server.run(transport="stdio")
