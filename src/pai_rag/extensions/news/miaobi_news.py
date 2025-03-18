import traceback
from typing import Dict, List

from pai_rag.app.api.models import ChatIntentType, ChatResponseWrapper
from pai_rag.extensions.news.news_config import MiaobiNewsConfig

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponseAsyncGen,
    ChatResponse,
    MessageRole,
)
from alibabacloud_aimiaobi20230801 import models as aimiaobi_models
from alibabacloud_aimiaobi20230801.client import Client as AimiaobiClient
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_openapi_sse.client import Client as OpenApiClient
from alibabacloud_tea_openapi_sse import models as open_api_models
from alibabacloud_tea_util_sse import models as open_api_util_models
import json

from pydantic import BaseModel
from loguru import logger

from pai_rag.integrations.llms.pai.pai_llm import PaiLlm


DEFAULT_NEWS_ERROR_MESSAGE = "抱歉，查询新闻发生错误，请稍后重试。"


def _create_client(
    access_key_id: str,
    access_key_secret: str,
    endpoint: str,
) -> OpenApiClient:
    config = open_api_models.Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        endpoint=endpoint,
    )
    return OpenApiClient(config)


class LightApp:
    def __init__(
        self, access_key_id, access_key_secret, endpoint, workspace_id, action
    ) -> None:
        # 工程代码泄露可能会导致 AccessKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考。
        # 建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html。
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.workspace_id = workspace_id
        if endpoint is None:
            endpoint = "quanmiaolightapp.cn-beijing.aliyuncs.com"
        self.endpoint = endpoint

        self._runtime = open_api_util_models.RuntimeOptions(read_timeout=1000 * 100)
        self._action = action

        self._api_info = self._create_api_info()

        self._client = _create_client(
            self.access_key_id, self.access_key_secret, self.endpoint
        )

    def _create_api_info(self) -> open_api_models.Params:
        """
        API 相关
        @param path: params
        @return: OpenApi.Params
        """
        params = open_api_models.Params(
            # 接口名称
            action=self._action,
            # 接口版本
            version="2024-08-01",
            # 接口协议
            protocol="HTTPS",
            # 接口 HTTP 方法
            method="POST",
            auth_type="AK",
            style="RPC",
            # 接口 PATH
            pathname=f"/{self.workspace_id}/quanmiao/lightapp/{self._action[0].lower() + self._action[1:]}",
            # 接口请求体内容格式,
            req_body_type="formData",
            # 接口响应体内容格式,
            body_type="sse",
        )
        return params

    async def do_sse_query(self, body):
        request = open_api_models.OpenApiRequest(body=body)
        sse_receiver = self._client.call_sse_api_async(
            params=self._api_info, request=request, runtime=self._runtime
        )
        return sse_receiver


def create_light_app_client(config: MiaobiNewsConfig):
    assert config.is_enabled(), "Miaobi News configuration must be provided."
    return LightApp(
        access_key_id=config.access_key_id,
        access_key_secret=config.access_key_secret,
        endpoint=config.endpoint,
        workspace_id=config.workspace_id,
        action="RunHotTopicChat",
    )


def create_aimiaobi_client(config: MiaobiNewsConfig):
    return AimiaobiClient(
        Config(
            access_key_id=config.access_key_id,
            access_key_secret=config.access_key_secret,
            endpoint="aimiaobi.cn-beijing.aliyuncs.com",
        )
    )


def _transform_messages(messages: List[ChatMessage]):
    return [
        {"role": message.role.value, "content": message.content} for message in messages
    ]


def _make_context(topics):
    return "\n\n".join(
        [f"标题: {topic['title']}\n摘要: {topic['summary']}" for topic in topics]
    )


class NewsChatParameter(BaseModel):
    workspaceId: str
    messages: List[Dict[str, str]] = []
    prompt: str = None


DEFAULT_PROMPT_TEMPLATE = """
你是一个专业的新闻播报员，负责整理每天的热点资讯列表并广播给车机端的用户。

# 【人设风格】
风格亲切、自然但不失专业性的新闻女主播

# 【热点新闻列表】
{hot_topics_str}

# 【输出格式】
- 请根据上下文信息，不要使用其他信息，参考【人设风格】，结构条理化的播放热点资讯。
- 注意每条新闻播报不要超过100个字。
"""


class MiaobiNewsTool:
    def __init__(self, llm: PaiLlm, config: MiaobiNewsConfig):
        self.llm = llm
        self.config = config
        self.chat_client = create_light_app_client(config)
        self.miaobi_client = create_aimiaobi_client(config)
        logger.info(
            f"MiaobiNewsTool initialized with workspace_id {config.workspace_id}."
        )

    async def _alist_hot_topics(self):
        request = aimiaobi_models.GetHotTopicBroadcastRequest(
            workspace_id=self.config.workspace_id,
            size=self.config.top_news_count,
            current=1,
        )

        broadcast_response = await self.miaobi_client.get_hot_topic_broadcast_async(
            request=request
        )
        assert (
            broadcast_response.status_code == 200
        ), "Get hot topic status code is not 200."
        hot_topics = []
        for topic in broadcast_response.body.data.data:
            hot_topics.append(
                {
                    "title": topic.news[0].title,
                    "url": topic.news[0].url,
                    "summary": topic.news[0].summary,
                    "category": topic.category,
                }
            )

        return hot_topics

    async def alist_topics(
        self,
    ) -> ChatResponseWrapper:
        try:
            hot_topics = await self._alist_hot_topics()
        except Exception as ex:
            logger.error(
                f"List news api failed. Exception: {ex}. {traceback.format_exc()}"
            )
            response = ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=DEFAULT_NEWS_ERROR_MESSAGE,
                ),
                delta=DEFAULT_NEWS_ERROR_MESSAGE,
                additional_kwargs={"news_articles": []},
            )
            return ChatResponseWrapper(response=response)

        try:
            messages = [
                ChatMessage(
                    role="user",
                    content=DEFAULT_PROMPT_TEMPLATE.format(
                        hot_topics_str=_make_context(hot_topics)
                    ),
                )
            ]

            response = await self.llm.achat(messages)
            response.additional_kwargs["news_articles"] = hot_topics
            return ChatResponseWrapper(response=response)
        except Exception as ex:
            logger.error(
                f"News chat llm failed. Exception: {ex}. {traceback.format_exc()}"
            )
            raise ex

    async def astream_list_topics(
        self,
    ) -> ChatResponseWrapper:
        try:

            async def gen() -> ChatResponseAsyncGen:
                yield ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        content="",
                    ),
                    delta="",
                    intent=ChatIntentType.LIST_NEWS,
                    additional_kwargs={"intent": ChatIntentType.LIST_NEWS},
                )

                try:
                    hot_topics = await self._alist_hot_topics()
                except Exception as ex:
                    logger.error(
                        f"List news api failed. Exception: {ex}. {traceback.format_exc()}"
                    )
                    yield ChatResponse(
                        message=ChatMessage(
                            role="assistant",
                            content=DEFAULT_NEWS_ERROR_MESSAGE,
                        ),
                        delta=DEFAULT_NEWS_ERROR_MESSAGE,
                        additional_kwargs={"news_articles": []},
                    )
                    return

                messages = [
                    ChatMessage(
                        role="user",
                        content=DEFAULT_PROMPT_TEMPLATE.format(
                            hot_topics_str=_make_context(hot_topics)
                        ),
                    )
                ]
                yield ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        content="",
                    ),
                    delta="",
                    additional_kwargs={"news_articles": hot_topics},
                )

                async for response in await self.llm.astream_chat(
                    messages=messages,
                ):
                    yield response

            return ChatResponseWrapper(response=gen())
        except Exception as e:
            logger.error(
                f"Error while getting hot topics: {e}, {traceback.format_exc()}"
            )
            raise e

    async def achat(
        self,
        prompt: str,
        messages: List[ChatMessage] = [],
    ):
        stream_response_wrapper = await self.astream_chat(
            prompt=prompt,
            messages=messages,
        )
        message_content = ""
        additional_kwargs = {}
        async for response in stream_response_wrapper.response:
            message_content += response.delta
            additional_kwargs.update(response.additional_kwargs)

        return ChatResponseWrapper(
            response=ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=message_content,
                ),
                additional_kwargs=additional_kwargs,
                source_nodes=stream_response_wrapper.source_nodes,
            )
        )

    async def astream_chat(
        self,
        prompt: str,
        messages: List[ChatMessage] = [],
    ) -> ChatResponseWrapper:
        logger.info(f"Chat news with prompt {prompt}, chat_history: {messages}")

        transformed_messages = _transform_messages(messages)
        param = NewsChatParameter(
            messages=transformed_messages,
            workspaceId=self.config.workspace_id,
            prompt=prompt,
        ).model_dump()

        async def gen() -> ChatResponseAsyncGen:
            origin_text = ""
            yield ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="",
                ),
                delta="",
                additional_kwargs={"intent": ChatIntentType.CHAT_NEWS},
            )

            async for item in await self.chat_client.do_sse_query(param):
                try:
                    data = json.loads(item.get("event").data)
                    logger.info(data)

                    event = data.get("header").get("event")
                    if event != "task-finished":
                        additional_kwargs = {}

                        usage = data.get("payload").get("usage")
                        if usage:
                            additional_kwargs.update(
                                {
                                    "completion_tokens": usage.get("outputTokens", 0),
                                    "prompt_tokens": usage.get("inputTokens", 0),
                                    "total_tokens": usage.get("totalTokens", 0),
                                }
                            )

                        search_query = (
                            data.get("payload").get("output").get("searchQuery")
                        )
                        if search_query:
                            additional_kwargs["search_query"] = search_query

                        hot_topics = (
                            data.get("payload").get("output").get("hotTopicSummaries")
                        )
                        if hot_topics:
                            news_articles = []
                            for topic in hot_topics:
                                news_articles.append(
                                    {
                                        "title": topic["news"][0]["title"],
                                        "url": topic["news"][0]["url"],
                                        "summary": topic["textSummary"],
                                    }
                                )
                            additional_kwargs["news_articles"] = news_articles

                        text = data.get("payload").get("output").get("text")
                        if text:
                            response = ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content=text[len(origin_text) :],
                                ),
                                delta=text[len(origin_text) :],
                                additional_kwargs=additional_kwargs,
                            )
                            origin_text = text
                            yield response

                        # 不只有usage信息
                        elif len(additional_kwargs) > 3:
                            empty_response = ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content="",
                                ),
                                delta="",
                                additional_kwargs=additional_kwargs,
                            )
                            yield empty_response

                except Exception as ex:
                    logger.warning(
                        f"Error when decoding Miaobi outputs {ex}, data: {item}"
                    )
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=DEFAULT_NEWS_ERROR_MESSAGE,
                        ),
                        delta=DEFAULT_NEWS_ERROR_MESSAGE,
                        additional_kwargs={},
                    )
                    continue

        return ChatResponseWrapper(response=gen())
