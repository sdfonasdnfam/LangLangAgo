"""One task, one event loop, one persistent MCP session."""

import asyncio
import os
import shutil
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from browser_agent.browser_tools import page_reader
from browser_agent.graph import build_graph
from browser_agent.skills import load_skills

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    api_key: str
    base_url: str
    model: str
    max_steps: int = 15
    headless: bool = True
    timeout: int = 180
    executable_path: str = ""

    def __post_init__(self):
        if not self.api_key.strip():
            raise ValueError("请填写 API Key")
        url = urlparse(self.base_url)
        if url.scheme not in {"https", "http"} or not url.netloc:
            raise ValueError("模型 Base URL 必须是有效的 HTTP(S) URL")
        if not self.model.strip():
            raise ValueError("请填写模型名称")
        if not 1 <= self.max_steps <= 50:
            raise ValueError("模型轮数必须在 1 到 50 之间")
        if not 10 <= self.timeout <= 600:
            raise ValueError("任务超时必须在 10 到 600 秒之间")


def browser_server(headless: bool = True, executable_path: str = ""):
    cli = ROOT / "node_modules" / "@playwright" / "mcp" / "cli.js"
    if not cli.is_file():
        raise FileNotFoundError("未安装 Playwright MCP，请先运行 ppnpm install")
    node = shutil.which("node")
    if not node:
        raise FileNotFoundError("未找到 Node.js，请先安装 Node.js 18 或更新版本")
    args = [str(cli), "--browser", "chromium", "--isolated", "--codegen", "none"]
    if headless:
        args.append("--headless")
    if executable_path:
        args.extend(["--executable-path", executable_path])
    # MCP does not need the model credential. Pass only process/browser environment.
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        in {
            "PATH",
            "HOME",
            "USERPROFILE",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "XDG_RUNTIME_DIR",
            "XDG_CACHE_HOME",
            "PLAYWRIGHT_BROWSERS_PATH",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        }
    }
    return StdioServerParameters(command=node, args=args, env=env)


@asynccontextmanager
async def connect_tools(server: StdioServerParameters):
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield await load_mcp_tools(session)


async def run_task(task: str, settings: Settings, on_update: Callable | None = None) -> dict:
    result = {
        "task": task,
        "status": "running",
        "answer": "",
        "steps": 0,
        "selected_skills": [],
        "messages": [],
    }
    try:
        if not task.strip():
            raise ValueError("任务不能为空")
        async with asyncio.timeout(settings.timeout):
            server = browser_server(settings.headless, settings.executable_path)
            async with connect_tools(server) as tools:
                model = ChatOpenAI(
                    api_key=settings.api_key,
                    base_url=settings.base_url,
                    model=settings.model,
                    temperature=0,
                    timeout=60,
                    max_retries=1,
                )
                graph = build_graph(
                    model, [page_reader(tools), *tools], load_skills(ROOT / "skills")
                )
                async for update in graph.astream(
                    {"task": task, "max_steps": settings.max_steps},
                    config={"recursion_limit": settings.max_steps * 2 + 5},
                    stream_mode="updates",
                ):
                    for node, values in update.items():
                        result.update({k: v for k, v in values.items() if k != "messages"})
                        result["messages"].extend(values.get("messages", []))
                        if on_update:
                            on_update({"node": node, **values})
    except Exception as exc:
        # This is the application boundary; programmer errors remain failures, not success.
        def describe(error):
            if isinstance(error, BaseExceptionGroup):
                return "; ".join(describe(e) for e in error.exceptions)
            if isinstance(error, TimeoutError):
                return "任务超时，浏览器会话已结束。"
            return f"{type(error).__name__}: {error}"

        result.update(status="failed", answer=describe(exc).replace(settings.api_key, "***"))
        if on_update:
            on_update({"node": "error", "status": "failed", "answer": result["answer"]})
    return result
