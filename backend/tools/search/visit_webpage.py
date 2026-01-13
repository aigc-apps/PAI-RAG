import json
import re
import aiohttp
import time
import asyncio
import tiktoken
from functools import partial
from typing import List, Union, Annotated
from llama_index.core.tools import FunctionTool
from loguru import logger
from utils.http_session import HttpSessionShared

TRUNCATE_TOKEN_MAXLENGTH = 5000

def truncate_to_tokens(text: str, max_tokens: int = TRUNCATE_TOKEN_MAXLENGTH) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        logger.info(f"Current tokens length is {len(tokens)}, return original text.")
        return text
    logger.info(f"Current tokens length is {len(tokens)}, truncating to {max_tokens} tokens.")
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens) + " \n\n [truncated] The content is too long, has been truncated."

def remove_images_and_links(text: str) -> str:
    # 1. 移除 Markdown 图片: ![...](...)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # 2. 清理多余空行和空白
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

async def jina_readpage(url: str) -> str:
    """使用 Jina Reader 读取网页内容"""
    max_retries = 3
    timeout = aiohttp.ClientTimeout(total=50)

    for attempt in range(max_retries):
        try:
            session = await HttpSessionShared.ensure_session()
            async with session.get(f"https://r.jina.ai/{url.strip()}", timeout=timeout) as response:
                text = await response.text()
                if response.status == 200:
                    return remove_images_and_links(text)
                else:
                    logger.warning(f"Jina 返回非200状态码: {response.status} - {text}")
        except Exception as e:
            logger.warning(f"Jina 请求失败 (尝试 {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return "[visit] Failed to read page."
            time.sleep(0.5)
    return "[visit] Failed to read page."

async def readpage_and_truncate(url: str, goal: str) -> dict:
    """读取网页并截断过长内容"""
    content = await jina_readpage(url)
    logger.info(f"Jina 读取网页成功, 内容长度 {len(content.strip())}")
    if not content or content.startswith("[visit] Failed to read page."):
        return {
            "url": url,
            "goal": goal,
            "content": "The provided webpage content could not be accessed. Please check the URL or file format.",
            "success": False
        }

    # 截断内容
    content = truncate_to_tokens(content, max_tokens=TRUNCATE_TOKEN_MAXLENGTH)
    return {
        "url": url,
        "goal": goal,
        "content": content,
        "success": True
    }

async def avisit_webpage(
    url: Union[str, List[str]],
    goal: str
) -> str:
    """
    访问一个或多个网页并返回结构化摘要。
    """
    start_time = time.time()
    max_time = 900  # 15分钟超时

    if isinstance(url, str):
        url_list = [url]
    elif isinstance(url, list):
        url_list = url
    else:
        raise ValueError("url 参数必须是字符串或列表")

    results = []
    for u in url_list:
        if time.time() - start_time > max_time:
            results.append({
                "url": u,
                "goal": goal,
                "content": "No content available due to timeout.",
                "success": False
            })
            continue

        try:
            result = await readpage_and_truncate(u, goal)
            results.append(result)
        except Exception as e:
            logger.error(f"处理 {u} 时出错: {e}")
            results.append({
                "url": u,
                "goal": goal,
                "content": "Processing failed.",
                "success": False
            })

    # 如果只有一个 URL，直接返回对象；多个则返回列表
    if len(results) == 1:
        output = results[0]
    else:
        output = {"results": results}

    return json.dumps(output, ensure_ascii=False, indent=2)

async def avisit_webpage_tool(url: Union[str, List[str]], goal: str):
    """Async visit webpage tool entry"""
    try:
        content = await avisit_webpage(url=url, goal=goal)
        return content
    except Exception as e:
        logger.error(f"Webpage visit tool failed: {e}")
        return json.dumps({
            "error": str(e),
            "url": url,
            "goal": goal
        }, ensure_ascii=False)

async def aget_visit_webpage_tool():
    """
    Visit webpage(s) and return the content.
    """
    avisit_webpage_tool_func = partial(avisit_webpage_tool)

    async def visit_webpage_handler(
        url: Annotated[
            str | List[str],
            "the URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs",
        ] = "",
        goal: Annotated[
            str,
            "the goal of the visit for webpage(s)",
        ] = "",
        **kwargs
    ):
        logger.info(
            f"visit_webpage_handler with url: {url}, goal: {goal}"
        )
        return await avisit_webpage_tool_func(
            url=url,
            goal=goal
        )

    visit_tool = FunctionTool.from_defaults(
        async_fn=visit_webpage_handler,
        name="visit_webpage",
        description="""Visit webpage(s) and return the content of webpage(s).
Params:
- url: required, string | list[string], the URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.
- goal: required, string, the goal of the visit for webpage(s).
Returns:
- {
  "results": [
    {
      "url": "string",
      "goal": "string",
      "content": "string",
      "success": true
    }
  ]
}
""",
        return_direct=False,
    )
    return visit_tool


if __name__ == "__main__":
    print(asyncio.run(avisit_webpage_tool(url = "https://www.qweather.com/weather30d/hangzhou-101210101.html", goal="确定下个月杭州到上海的天气情况")))
    print(asyncio.run(avisit_webpage_tool( url =["https://www.klook.com/zh-CN/china-high-speed-rail/19190-hangzhou/59-shanghai/","https://tw.trip.com/trains/china/route/hangzhou-to-shanghai/", "https://trains.ctrip.com/trainbooking/hangzhou-shanghai/gaotie"], goal="查询从杭州到上海的往返高铁时刻表和票价信息，重点关注早上从杭州出发和晚上从上海返回的班次。")))
