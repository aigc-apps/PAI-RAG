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
ACCEPATABLE_NEWS_TOPICS = set(["科技", "娱乐", "经济", "时政", "社会", "体育", "教育", "国际"])


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
    return "\n".join(
        [
            f"【新闻 {i+1}】. {topic['title']}\n{topic['summary']}\n"
            for i, topic in enumerate(topics)
        ]
    )


DEFAULT_PROMPT_TEMPLATE = """
# 【任务描述】你是深小闻，是一个车机新闻播报小助手。你会根据下面给出的新闻材料，按顺序有条理的播报所有新闻。

# 【人设风格】风格亲切、自然但不失专业性的新闻主播深小闻。

# 【精选{topics_str}新闻列表】
{news_list_str}

# 【输出格式】
- 简短、友好的开场导语，如深小闻为您带来今天的热点资讯、深小闻为你推荐下面的科技热点等。
- 保持亲切、自然的语言风格同时不失专业性。
- 请根据新闻列表中信息播报，不要使用其他信息。
- 请遵循新闻给出的顺序，结构化、有条理的归纳每条新闻内容并用数字序号标识。
- 注意每条新闻播报不要超过100个字。
"""

DEFAULT_CHAT_CUSTOM_PROMPT_TEMPLATE = """
# 【任务描述】你是深小闻，一个智能车机问答助手，你的职责是根据给定的上下文信息回答问题。

# 【上下文信息】
{content}

# 【人设风格】风格亲切、自然但不失专业性的新闻主播深小闻

# 【输出格式】
- 请根据上下文信息，不要使用其他信息，参考【人设风格】，结构条理化的回答问题“{prompt}”。
- 回答时使用简短、友好的开场导语，如深小闻为您带来关于（）的热点新闻.
- 内容的字数一定控制在{answerLength}个字符以内。
- 如果不能回答，请输出：根据已知信息无法回答。
"""


class NewsChatParameter(BaseModel):
    workspaceId: str
    messages: List[Dict[str, str]] = []
    prompt: str = None
    modelCustomPromptTemplate: str = DEFAULT_CHAT_CUSTOM_PROMPT_TEMPLATE
    answerLength: int = 200


class MiaobiNewsTool:
    def __init__(self, llm: PaiLlm, config: MiaobiNewsConfig):
        self.llm = llm
        self.config = config
        self.chat_client = create_light_app_client(config)
        self.miaobi_client = create_aimiaobi_client(config)
        logger.info(
            f"MiaobiNewsTool initialized with workspace_id {config.workspace_id}."
        )

    async def _alist_hot_topics(self, news_topics):
        request = aimiaobi_models.GetHotTopicBroadcastRequest(
            workspace_id=self.config.workspace_id,
            size=self.config.top_news_count,
            current=1,
            step_for_news_broadcast_content_config=aimiaobi_models.GetHotTopicBroadcastRequestStepForNewsBroadcastContentConfig(
                categories=news_topics
            ),
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
        query_str: str,
        news_topics: List[str] = [],
    ) -> ChatResponseWrapper:
        try:
            hot_topics = await self._alist_hot_topics(news_topics=news_topics)
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
            content = DEFAULT_PROMPT_TEMPLATE.format(
                news_list_str=_make_context(hot_topics),
                query_str=query_str,
                topics_str="、".join(news_topics),
            )
            messages = [
                ChatMessage(
                    role="user",
                    content=content,
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
        query_str: str,
        news_topics: List[str] = [],
    ) -> ChatResponseWrapper:
        try:

            async def gen() -> ChatResponseAsyncGen:
                yield ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        content="",
                    ),
                    delta="",
                    additional_kwargs={
                        "intent": ChatIntentType.LIST_NEWS,
                        "news_topics": news_topics,
                    },
                )

                try:
                    hot_topics = await self._alist_hot_topics(news_topics=news_topics)
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
                            news_list_str=_make_context(hot_topics),
                            query_str=query_str,
                            topics_str="、".join(news_topics),
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
            logger.info(f" Chat news with param {param}.")
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
