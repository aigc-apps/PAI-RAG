import asyncio
from typing import List
from common.tool.search_result import SearchResult
from loguru import logger

from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_iqs20241111 import models
from alibabacloud_iqs20241111.client import Client
import time


DEFAULT_ALIYUN_SEARCH_ENDPOINT = "iqs.cn-zhangjiakou.aliyuncs.com"
DEFAULT_SEARCH_QA_PROMPT_TEMPLATE = """
你的目标是根据搜索结果提供准确、有用且易于理解的信息。
# 任务要求：
- 请严格根据提供的参考内容回答问题，并非所有参考内容都与用户的问题密切相关，你需要结合问题，对参考内容进行甄别、筛选。仅参考与问题相关的内容并忽略所有不相关的信息。
- 如果参考内容中没有相关信息或与问题无关，请基于你的已有知识进行回答。
- 确保答案准确、简洁，并且使用与用户提问相同的语种。
- 在回答过程中，请避免使用“从参考内容得出”、“从材料得出”、“根据参考内容”等措辞。
- 保持回答的专业性和友好性。
- 如果需要更多信息来更好地回答问题，请礼貌地询问。
- 对于复杂的问题，尽量简化解释，使信息易于理解。如果回答很长，请尽量结构化、分段落总结。如果需要分点作答，尽量控制在5个点以内，并合并相关的内容。
- 对于客观类的问答，如果问题的答案非常简短，可以适当补充一到两句相关信息，以丰富内容。
- 除非用户要求，否则请保持输出语种与用户输入问题语种的一致性。
- 对于涉及不安全/不道德/敏感/色情/暴力/赌博/违法等行为的问题，请明确拒绝提供所要求的信息，并简单解释为什么这样的请求不能被满足。
"""

DEFAULT_SEARCH_COUNT = 10
DEFAULT_LANG = "zh-CN"
DEFAULT_TIMERANGE = "OneMonth"  # OneMonth, OneWeek, OneDay, OneYear, NoLimit


class NodeWithScore:
    def __init__(self, text, score, metadata):
        self.text = text
        self.score = score
        self.metadata = metadata

    def to_dict(self):
        return {"text": self.text, "metadata": self.metadata, "score": self.score}


class AliyunSearchTool:
    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        endpoint: str = DEFAULT_ALIYUN_SEARCH_ENDPOINT,
        search_count: int = DEFAULT_SEARCH_COUNT,
        search_lang: str = DEFAULT_LANG,
        time_range: str = DEFAULT_TIMERANGE,
        search_qa_prompt_template: str = DEFAULT_SEARCH_QA_PROMPT_TEMPLATE,
    ):
        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
        )

        self.search_count = search_count
        self.search_lang = search_lang

        config.endpoint = endpoint
        self.Client = Client(config)
        self.time_range = time_range
        self.search_qa_prompt_template = search_qa_prompt_template

    async def _search_aliyun_single_page(self, query: str, page: int = 1):
        request = models.GenericSearchRequest(
            query=query,
            time_range=self.time_range,
            page=page,
        )
        response = await self.Client.generic_search_async(request)
        if response.status_code != 200:
            logger.warning(
                f"Aliyun Search API failed, status code {response.status_code}, detail {response}"
            )
            return []
        logger.info(f"Finished searching query {request}.")
        return response.body.to_map()

    async def _asearch(
        self,
        query: str,
    ) -> List[SearchResult]:
        search_tasks = []
        for i in range(0, 1 + int((self.search_count - 1) / 10), 1):
            search_tasks.append(
                self._search_aliyun_single_page(query=query, page=i + 1)
            )

        search_results = await asyncio.gather(*search_tasks)
        results = []
        for result in search_results:
            items = result.get("pageItems")
            for item in items:
                text = (
                    item.get("markdownText")
                    or item.get("mainText")
                    or item.get("htmlSnippet")
                )
                if not text:
                    continue

                score = 0.1
                if item.get("score"):
                    score = item.get("score")
                host_logo = "https://cdn.pixabay.com/photo/2020/09/17/22/52/website-5580513_1280.png"
                if item.get("hostLogo") and item.get("hostLogo") != "":
                    host_logo = item.get("hostLogo")
                results.append(
                    SearchResult(
                        content=text[:1000],
                        url=item.get("link"),
                        title=item.get("title") or item.get("htmlTitle"),
                        hostname=item.get("hostname"),
                        favicon=host_logo,
                        publish_time=str(item.get("publishTime")),
                        score=score,
                    )
                )
                if len(results) >= self.search_count:
                    break
        return results


    async def aquery(
        self,
        query: str,
    ) -> List[dict]:
        start = time.time()
        logger.info(f"Aliyun Search with query {query}.")
        nodes = await self._asearch(query=query)
        logger.info(
            f"[WebSearch]-Aliyun: Get {len(nodes)} docs from url. Elapsed time: {time.time() - start}seconds."
        )

        return {"result": [node.model_dump() for node in nodes]}
