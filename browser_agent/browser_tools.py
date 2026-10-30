"""Combine navigation and reading so research does not spend a model turn on each."""

from langchain_core.tools import BaseTool, tool


def page_reader(tools: list[BaseTool]) -> BaseTool:
    browser = {t.name: t for t in tools}

    @tool
    async def browser_read_page(url: str) -> str | list:
        """Open a URL and read its title, final URL, main text and links in one call.

        Prefer this for researching documentation or collecting information.
        For clicking or filling fields, use browser_snapshot to obtain element refs.
        Returns bounded text; use browser_evaluate to extract specific missing sections.
        """
        await browser["browser_navigate"].ainvoke({"url": url})
        return await browser["browser_evaluate"].ainvoke(
            {
                "function": """() => {
                const main = document.querySelector('article')
                    || document.querySelector('main') || document.body;
                const text = main.innerText;
                const links = [...main.querySelectorAll('a[href]')]
                    .filter(a => a.innerText.trim() && /^https?:/.test(a.href))
                    .map(a => ({title: a.innerText.trim(), url: a.href}));
                return {title: document.title, url: location.href,
                    text: text.slice(0, 10000), truncated: text.length > 10000,
                    links: links.slice(0, 30), linksTruncated: links.length > 30};
            }"""
            }
        )

    return browser_read_page
