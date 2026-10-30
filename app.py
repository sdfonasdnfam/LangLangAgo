"""Run with: uv run streamlit run app.py"""

import asyncio
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from browser_agent.runtime import ROOT, Settings, run_task
from browser_agent.skills import load_skills

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="浏览器 Agent", page_icon="🌐", layout="wide")
st.title("🌐 浏览器 Agent")
st.caption("LangGraph · Playwright MCP · Skill | 输入任务，观察执行过程，查看结果")

with st.sidebar:
    st.header("模型与浏览器")
    api_key = st.text_input("API Key", value=os.getenv("LLM_API_KEY", ""), type="password")
    base_url = st.text_input(
        "Base URL", value=os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    )
    model = st.text_input("模型名称", value=os.getenv("LLM_MODEL", "deepseek-chat"))
    max_steps = st.number_input("最大模型轮数（含最后整理）", min_value=1, max_value=50, value=15)
    timeout = st.number_input("任务超时（秒）", min_value=10, max_value=600, value=180)
    headless = st.checkbox("无头浏览器", value=True)
    executable = st.text_input(
        "浏览器可执行文件（可选）", value=os.getenv("BROWSER_EXECUTABLE_PATH", "")
    )
    st.caption("模型须支持工具调用。每个任务使用独立浏览器会话，结束后自动关闭。")
    with st.expander("可用 Skill"):
        try:
            for skill in load_skills(ROOT / "skills"):
                st.markdown(f"**{skill.name}**：{skill.description}")
                st.caption("关键词：" + "、".join(skill.keywords))
        except ValueError as exc:
            st.error(str(exc))

with st.form("task_form"):
    task = st.text_area(
        "浏览器任务", placeholder="例如：打开 https://example.com，提取页面标题和正文。"
    )
    submitted = st.form_submit_button("开始执行", type="primary")

if submitted:
    if not task.strip():
        st.warning("请先输入任务。")
    else:
        try:
            settings = Settings(api_key, base_url, model, max_steps, headless, timeout, executable)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state.pop("result", None)
            st.session_state["events"] = []
            with st.status("正在连接 MCP 并执行任务…", expanded=True) as progress:

                def on_update(update):
                    st.session_state["events"].append(update)
                    if update["node"] == "skills":
                        st.write(
                            "加载 Skill："
                            + ("、".join(update["selected_skills"]) or "通用浏览器策略")
                        )
                    elif update["node"] == "agent":
                        st.write(f"模型决策：第 {update['steps']} 轮")
                        if update.get("stop_reason"):
                            st.write(update["stop_reason"])
                    elif update["node"] == "tools":
                        st.write("浏览器工具已返回结果")

                result = asyncio.run(run_task(task, settings, on_update))
                st.session_state["result"] = result
                progress.update(
                    label="任务执行结束",
                    expanded=False,
                    state="error" if result["status"] == "failed" else "complete",
                )

if "result" in st.session_state:
    result = st.session_state["result"]
    labels = {
        "completed": "模型已结束",
        "stopped": "已停止探索，以下是当前结果",
        "failed": "执行失败",
    }
    st.subheader(labels[result["status"]])
    st.caption(
        f"模型调用 {result['steps']} 轮 · Skill："
        + ("、".join(result["selected_skills"]) or "通用策略")
    )
    st.caption(
        f"累计输入 {result.get('input_tokens', 0):,} tokens · "
        f"累计输出 {result.get('output_tokens', 0):,} tokens（以接口返回用量为准）"
    )
    st.markdown(result["answer"])
    with st.expander("执行记录", expanded=result["status"] != "completed"):
        for event in st.session_state["events"]:
            for message in event.get("messages", []):
                if isinstance(message, AIMessage):
                    if message.text:
                        st.markdown(message.text)
                    for call in message.tool_calls:
                        st.write(f"调用：{call['name']}")
                        st.json(call["args"])
                elif isinstance(message, ToolMessage):
                    st.write(f"返回：{message.name or message.tool_call_id} · {message.status}")
                    st.code(
                        message.content
                        if isinstance(message.content, str)
                        else str(message.content),
                        language=None,
                    )
