import asyncio
import time
from pairag.integrations.search.quark_utils import (
    get_access_token,
    postprocess_items,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import QueryBundle
from llama_index.core.query_engine import BaseQueryEngine
from urllib.parse import urljoin, urlencode
import httpx
from loguru import logger

from pairag.integrations.search.search_config import (
    DEFAULT_SEARCH_COUNT,
    DEFAULT_SEARCH_QA_PROMPT_TEMPLATE,
)


class QuarkAccessTokenProvider:
    def __init__(self, host: str, user: str, secret: str):
        self.host = host
        self.user = user
        self.secret = secret
        self.token = None
        self.expire_at = 0

    async def _refresh_token(self):
        self.expire_at = time.time()
        self.token, valid_duration = await get_access_token(
            self.host, self.user, self.secret
        )
        self.expire_at += valid_duration
        logger.info(f"Refreshed Quark token {self.token}, expires at {self.expire_at}.")

    async def get_token(self):
        # 5 minutes buffer
        if self.token and self.expire_at > time.time() + 300:
            return self.token

        await self._refresh_token()
        return self.token


class QuarkSearchTool(BaseQueryEngine):
    def __init__(
        self,
        user: str,
        secret: str,
        host: str,
        synthesizer: BaseSynthesizer = None,
        search_count: int = DEFAULT_SEARCH_COUNT,
        search_qa_prompt_template: str = DEFAULT_SEARCH_QA_PROMPT_TEMPLATE,
    ):
        self.host = host
        self.user = user
        self.secret = secret

        self.token_provider = QuarkAccessTokenProvider(host, user, secret)
        self.synthesizer = synthesizer
        self.search_count = search_count
        self.search_qa_prompt_template = search_qa_prompt_template

    async def _search_quark_single_page(self, query: str, token: str, page: int = 1):
        async with httpx.AsyncClient() as client:
            payload = {
                "q": query,
                "page": page,
                "fill_item": 0,
            }
            search_url = urljoin(
                self.host, f"/api/resource/s_agg/ex/query?{urlencode(payload)}"
            )
            response = await client.post(
                search_url,
                headers={"Authorization": f"Bearer {token}"},
            )

            logger.info(f"Finished searching query {payload}. {response.json()}")
            return response.json()

    async def asearch(self, query: str):
        search_tasks = []
        token = await self.token_provider.get_token()
        for i in range(0, 1 + int((self.search_count - 1) / 10), 1):
            search_tasks.append(
                self._search_quark_single_page(query=query, token=token, page=i + 1)
            )

        search_results = await asyncio.gather(*search_tasks)

        nodes = []
        for result in search_results:
            items = postprocess_items(result["items"]["item"])
            for item in items:
                node = TextNode(
                    text=item["text"][:800],
                    metadata={"file_url": item["url"], "file_name": item["title"]},
                )
                if item.get("time"):
                    node.metadata["publish_time"] = item["time"]
                if item.get("source"):
                    node.metadata["source"] = item["source"]
                nodes.append(NodeWithScore(node=node, score=1))
                if len(nodes) >= self.search_count:
                    break
        return nodes

    async def aquery(self, query: QueryBundle):
        start = time.time()

        logger.info(f"Quark Search with query {query.query_str}.")
        nodes = await self.asearch(query=query.query_str)
        logger.info(
            f"[WebSearch]-Quark: Get {len(nodes)} docs from url. Elapsed time: {time.time() - start}seconds."
        )

        return await self.synthesizer.asynthesize(
            query=query,
            nodes=nodes,
            system_role_str=" ",
            prompt_template_str=self.search_qa_prompt_template,
            **query.llm_kwargs,
        )

    def _get_prompt_modules(self):
        raise NotImplementedError

    def _query(self, query_bundle):
        raise NotImplementedError

    async def _aquery(self, query_bundle):
        raise NotImplementedError
