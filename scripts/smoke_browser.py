"""Real Playwright MCP smoke test against a local page; no LLM/API key."""

import asyncio
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from browser_agent.runtime import ROOT, browser_server, connect_tools  # noqa: E402

PAGE = b"""<!doctype html><html><head><title>MCP Course Demo</title></head>
<body><h1>Browser smoke test</h1><label>Name <input aria-label="Name"></label>
<button onclick="document.querySelector('output').textContent=
'Hello '+document.querySelector('input').value">Greet</button>
<output></output></body></html>"""


class PageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, *args):
        pass


async def smoke(url):
    async with asyncio.timeout(60):
        async with connect_tools(
            browser_server(executable_path=os.getenv("BROWSER_EXECUTABLE_PATH", ""))
        ) as tools:
            available = {t.name: t for t in tools}
            page = await available["browser_navigate"].ainvoke({"url": url})
            if "MCP Course Demo" not in str(page):
                raise AssertionError(f"Navigation failed: {page}")
            # This smoke test isolates MCP/browser integration; graph decisions use scripted tests.
            result = await available["browser_run_code"].ainvoke(
                {
                    "code": """async (page) => {
                await page.getByRole('textbox', {name: 'Name'}).fill('Student');
                await page.getByRole('button', {name: 'Greet'}).click();
                return await page.locator('output').innerText();
            }"""
                }
            )
            if "Hello Student" not in str(result):
                raise AssertionError(f"Form interaction failed: {result}")
            print("PASS: MCP discovery → navigation → input → click → read result")


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    server = ThreadingHTTPServer(("127.0.0.1", 0), PageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        asyncio.run(smoke(f"http://127.0.0.1:{server.server_port}"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
