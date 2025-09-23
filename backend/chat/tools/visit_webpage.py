import json
import os
import requests
import time
import asyncio
import tiktoken
from functools import partial
from typing import List, Union, Annotated
from llama_index.core.tools import FunctionTool
from rag.chunk_helper import get_llm_from_db
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ChatResponse,
)
from loguru import logger

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
"""

@staticmethod
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


async def call_llm_for_summary(model:str, messages: List[ChatMessage], max_retries: int = 2) -> str:
    """调用 LLM 服务生成摘要"""
    llm = await get_llm_from_db(model_id=model)

    for attempt in range(max_retries):
        try:
            response: ChatResponse = llm.chat(messages)
            content = response.message.content.strip()

            if content:
                # 尝试提取 JSON 块
                left = content.find('{')
                right = content.rfind('}')
                if left != -1 and right != -1 and left <= right:
                    content = content[left:right+1]
            return content
        except Exception as e:
            logger.warning(f"LLM 调用失败，第 {attempt + 1} 次重试: {e}")
            if attempt == max_retries - 1:
                return ""
            await asyncio.sleep(1)
    return ""


async def jina_readpage(url: str) -> str:
    """使用 Jina Reader 读取网页内容"""
    max_retries = 3
    timeout = 50

    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"https://r.jina.ai/{url.strip()}",
                timeout=timeout
            )
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"Jina 返回非200状态码: {response.status_code} - {response.text}")
        except Exception as e:
            logger.warning(f"Jina 请求失败 (尝试 {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return "[visit] Failed to read page."
            time.sleep(0.5)
    return "[visit] Failed to read page."

async def readpage_and_summarize(model:str, url: str, goal: str) -> dict:
    """读取网页并生成结构化摘要"""
    content = await jina_readpage(url)

    if not content or content.startswith("[visit] Failed to read page."):
        return {
            "url": url,
            "goal": goal,
            "evidence": "The provided webpage content could not be accessed. Please check the URL or file format.",
            "summary": "The webpage content could not be processed, and therefore, no information is available.",
            "success": False
        }

    # 截断内容
    content = truncate_to_tokens(content, max_tokens=95000)
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[
                TextBlock(text=EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)),
            ],
        ),
    ]

    max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))
    summary_retries = 3
    raw = await call_llm_for_summary(model, messages, max_retries=max_retries)

    while len(raw) < 10 and summary_retries > 0:
        truncate_length = int(0.7 * len(content))
        logger.info(f"[visit] 摘要失败，截断至 {truncate_length} 字符，剩余重试 {summary_retries} 次")
        content = content[:truncate_length]
        messages[0]["content"] = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
        raw = await call_llm_for_summary(messages, max_retries=max_retries)
        summary_retries -= 1

    # 尝试解析 JSON
    parse_retry_times = 0
    while parse_retry_times < 3:
        try:
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            evidence = result.get("evidence", "")
            summary = result.get("summary", "")
            break
        except Exception as e:
            logger.warning(f"JSON 解析失败，第 {parse_retry_times + 1} 次重试: {e}")
            raw = await call_llm_for_summary(messages, max_retries=max_retries)
            parse_retry_times += 1
    else:
        evidence = "Failed to parse LLM response."
        summary = "No summary available due to processing error."

    return {
        "url": url,
        "goal": goal,
        "evidence": evidence,
        "summary": summary,
        "success": True
    }

async def avisit_webpage(
    model: str,
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
                "evidence": "Timeout: Processing exceeded 15 minutes.",
                "summary": "No summary available due to timeout.",
                "success": False
            })
            continue

        try:
            result = await readpage_and_summarize(model, u, goal)
            results.append(result)
        except Exception as e:
            logger.error(f"处理 {u} 时出错: {e}")
            results.append({
                "url": u,
                "goal": goal,
                "evidence": f"Error: {str(e)}",
                "summary": "Processing failed.",
                "success": False
            })

    # 如果只有一个 URL，直接返回对象；多个则返回列表
    if len(results) == 1:
        output = results[0]
    else:
        output = {"results": results}

    return json.dumps(output, ensure_ascii=False, indent=2)

async def avisit_webpage_tool(model:str, url: Union[str, List[str]], goal: str):
    """Async visit webpage tool entry"""
    try:
        content = await avisit_webpage(model=model, url=url, goal=goal)
        return content
    except Exception as e:
        logger.error(f"Webpage visit tool failed: {e}")
        return json.dumps({
            "error": str(e),
            "url": url,
            "goal": goal
        }, ensure_ascii=False)

async def aget_visit_webpage_tool(model: str):
    """
    Visit webpage(s) and return the summary of the content.
    """
    avisit_webpage_tool_func = partial(avisit_webpage_tool, model=model)

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
        description="""Visit webpage(s) and return the summary of the content.
Params:
- url: required, string | list[string], the URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.
- goal: required, string, the goal of the visit for webpage(s).
Returns:
- {
  "results": [
    {
      "url": "string",
      "goal": "string",
      "evidence": "string",
      "summary": "string",
      "success": true
    }
  ]
}
""",
    )
    return visit_tool


if __name__ == "__main__":
    print(asyncio.run(avisit_webpage_tool(model = "qwen-max", url =["https://www.klook.com/zh-CN/china-high-speed-rail/19190-hangzhou/59-shanghai/","https://tw.trip.com/trains/china/route/hangzhou-to-shanghai/", "https://trains.ctrip.com/trainbooking/hangzhou-shanghai/gaotie"], goal="查询从杭州到上海的往返高铁时刻表和票价信息，重点关注早上从杭州出发和晚上从上海返回的班次。")))
