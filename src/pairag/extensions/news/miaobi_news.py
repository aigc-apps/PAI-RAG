import traceback
from typing import Dict, List, Any, Sequence
from llama_index.core.prompts import PromptTemplate
from pairag.chat.models import ChatIntentType, ChatResponseWrapper
from pairag.extensions.news.news_config import (
    MiaobiNewsConfig,
    DEFAULT_NEWS_ROLE,
    DEFAULT_NEWS_ERROR_MESSAGE,
    DEFAULT_WEB_SEARCH_INFO_MESSAGE,
    DEFAULT_LIST_NEWS_END_RESPONSE,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponseAsyncGen,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    MessageRole,
    LLMMetadata,
)
from llama_index.core.base.response.schema import Response
from llama_index.core.instrumentation.events.query import QueryEndEvent

from llama_index.core.instrumentation.span import active_span_id

from llama_index.core.llms.llm import LLM
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback

from alibabacloud_aimiaobi20230801 import models as aimiaobi_models
from alibabacloud_aimiaobi20230801.client import Client as AimiaobiClient
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_openapi_sse.client import Client as OpenApiClient
from alibabacloud_tea_openapi_sse import models as open_api_models
from alibabacloud_tea_util_sse import models as open_api_util_models
import json

from openinference.instrumentation.llama_index import get_current_span
from pairag.integrations.trace.base import use_current_span
from pydantic import BaseModel
from loguru import logger

from pairag.integrations.llms.pai.pai_llm import PaiLlm
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


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


class NewsChatParameter(BaseModel):
    workspaceId: str
    messages: List[Dict[str, str]] = []
    prompt: str = None
    modelId: str = "qwen-max-latest"
    modelCustomPromptTemplate: str = None
    # answerLength: int = 200 # temporarily inactive


class MiaobiNewsTool(LLM):
    llm: PaiLlm = Field(description="")
    config: MiaobiNewsConfig = Field(description="")
    chat_client: LightApp = Field(description="")
    miaobi_client: AimiaobiClient = Field(description="")
    list_topics_prompt_template: PromptTemplate = Field(description="")
    chat_news_prompt_template: str = Field(description="")

    def __init__(self, llm: PaiLlm, config: MiaobiNewsConfig):
        chat_client = create_light_app_client(config)
        miaobi_client = create_aimiaobi_client(config)
        # self.chat_news_answer_len = config.chat_news_answer_len
        list_topics_prompt_template = PromptTemplate(
            template=config.list_topics_prompt_str
        )
        chat_news_prompt_template = config.chat_news_prompt_str.replace(
            "{news_role}", config.news_role
        )

        super().__init__(
            llm=llm,
            config=config,
            chat_client=chat_client,
            miaobi_client=miaobi_client,
            list_topics_prompt_template=list_topics_prompt_template,
            chat_news_prompt_template=chat_news_prompt_template,
        )

        logger.info(
            f"MiaobiNewsTool initialized with workspace_id {config.workspace_id}."
        )

    async def _alist_hot_topics(self, news_topics) -> List[Dict[str, Any]]:
        """Returns a list of dict, each dict represents a news, list is sorted in descending order of hot_value."""
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
                    "title": topic.hot_topic,
                    "url": topic.news[0].url,
                    "summary": topic.text_summary,
                    "category": topic.category,
                    "hot_value": topic.hot_value,
                }
            )
        sorted_hot_topics = sorted(
            hot_topics, key=lambda x: x["hot_value"], reverse=True
        )
        return sorted_hot_topics

    @dispatcher.span
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
                    role=MessageRole.ASSISTANT,
                    content=DEFAULT_NEWS_ERROR_MESSAGE,
                ),
                delta=DEFAULT_NEWS_ERROR_MESSAGE,
                additional_kwargs={"news_articles": []},
            )
            return ChatResponseWrapper(response=response)

        try:
            logger.debug(
                f"Using list_topics_prompt_template: {self.list_topics_prompt_template}"
            )
            content = self.list_topics_prompt_template.format(
                news_role=self.config.news_role,
                news_list_str=_make_context(hot_topics),
                query_str=query_str,
                topics_str="、".join(news_topics),
                conclusion_str=DEFAULT_LIST_NEWS_END_RESPONSE,
            )
            messages = [
                ChatMessage(
                    role="user",
                    content=content,
                )
            ]
            # store hot topics in span output
            span_id = active_span_id.get()
            dispatcher.event(
                QueryEndEvent(
                    response=Response(response=str(hot_topics), source_nodes=[]),
                    query="",
                    span_id=span_id,
                )
            )
            response = await self.llm.achat(messages)
            response.additional_kwargs["news_articles"] = hot_topics
            return ChatResponseWrapper(response=response)
        except Exception as ex:
            logger.error(
                f"News chat llm failed. Exception: {ex}. {traceback.format_exc()}"
            )
            raise ex

    @dispatcher.span
    async def astream_list_topics(
        self, messages: List[ChatMessage] = [], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        try:
            query_str = kwargs.get("query_str", "")
            news_topics = kwargs.get("news_topics", [])
            span_id = active_span_id.get()

            # use use_current_span decorator to keep miaobinews span
            # as the parent of the self.llm's span,
            # when self.llm.astream_chat executes in this gen()
            @use_current_span(get_current_span())
            async def gen() -> ChatResponseAsyncGen:
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
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
                            role=MessageRole.ASSISTANT,
                            content=DEFAULT_NEWS_ERROR_MESSAGE,
                        ),
                        delta=DEFAULT_NEWS_ERROR_MESSAGE,
                        additional_kwargs={"news_articles": []},
                    )
                    return

                logger.debug(
                    f"Using list_topics_prompt_template: {self.list_topics_prompt_template}"
                )
                messages = [
                    ChatMessage(
                        role="user",
                        content=self.list_topics_prompt_template.format(
                            news_role=self.config.news_role,
                            news_list_str=_make_context(hot_topics),
                            query_str=query_str,
                            topics_str="、".join(news_topics),
                            conclusion_str=DEFAULT_LIST_NEWS_END_RESPONSE,
                        ),
                    )
                ]
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="",
                    ),
                    delta="",
                    additional_kwargs={"news_articles": hot_topics},
                )

                # store hot topics in span output
                dispatcher.event(
                    QueryEndEvent(
                        response=Response(response=str(hot_topics), source_nodes=[]),
                        query="",
                        span_id=span_id,
                    )
                )
                async for response in await self.llm.astream_chat(
                    messages=messages,
                ):
                    yield response

            return gen()
        except Exception as e:
            logger.error(
                f"Error while getting hot topics: {e}, {traceback.format_exc()}"
            )
            raise e

    @llm_chat_callback()
    async def achat(
        self, messages: List[ChatMessage] = [], **kwargs: Any
    ) -> ChatResponse:
        args = {"prompt": kwargs.get("prompt", "")}
        stream_response_gen = await self.astream_chat(
            messages=messages,
            **args,
        )
        message_content = ""
        additional_kwargs = {}
        async for response in stream_response_gen:
            message_content += response.delta
            additional_kwargs.update(response.additional_kwargs)

        response = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=message_content,
            ),
            additional_kwargs=additional_kwargs,
        )
        return response

    @llm_chat_callback()
    async def astream_chat(
        self, messages: List[ChatMessage] = [], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        prompt = kwargs.get("prompt", "")
        logger.info(f"astream_chat with prompt {prompt}, chat_history: {messages}")

        transformed_messages = _transform_messages(messages)
        param = NewsChatParameter(
            messages=transformed_messages,
            workspaceId=self.config.workspace_id,
            prompt=prompt,
            modelId=self.config.chat_news_model_id,
            # answerLength=self.chat_news_answer_len,
            modelCustomPromptTemplate=self.chat_news_prompt_template,
        ).model_dump()

        async def gen() -> ChatResponseAsyncGen:
            origin_text = ""
            yield ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                ),
                delta="",
                additional_kwargs={"intent": ChatIntentType.CHAT_NEWS},
            )
            logger.info(f"Chat news with param {param}.")
            use_web_search = False
            additional_kwargs = {}
            async for item in await self.chat_client.do_sse_query(param):
                try:
                    data = json.loads(item.get("event").data)
                    logger.info(data)

                    event = data.get("header").get("event")
                    if event == "task-hot-topic-chat-internet-search-start":
                        use_web_search = True
                    if event != "task-finished" and event != "task-failed":
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

                        text = data.get("payload").get("output").get("text")
                        if text:
                            response = ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content=origin_text,
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
                    elif origin_text == "":
                        # (task finished or task failed) and no origin_text
                        text = data.get("payload").get("output").get(
                            "text"
                        ) or data.get("header").get("errorMessage")
                        err_code = data.get("header").get("errorCode")
                        logger.info(
                            f"News chat task-finished with err_code: {err_code}"
                        )
                        if text:
                            response = ChatResponse(
                                message=ChatMessage(
                                    role=MessageRole.ASSISTANT,
                                    content=origin_text,
                                ),
                                delta=text[len(origin_text) :],
                                additional_kwargs=additional_kwargs,
                            )
                            origin_text = text
                            yield response
                    else:
                        # (task finished or task failed) with origin_text
                        response = ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=origin_text,
                            ),
                            delta="",
                            additional_kwargs=additional_kwargs,
                        )
                        yield response
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

            if use_web_search:
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=f"{origin_text}\n{DEFAULT_WEB_SEARCH_INFO_MESSAGE}",
                    ),
                    delta=DEFAULT_WEB_SEARCH_INFO_MESSAGE,
                    additional_kwargs=additional_kwargs,
                )

        return gen()

    def _get_news_role_texts(self) -> List[str]:
        default_news_role_response = DEFAULT_NEWS_ROLE.format(
            domain_list="/".join(self.config.domain_list),
            news_role=self.config.news_role,
        )
        lines = []
        for line in default_news_role_response.split("\n"):
            if line.strip():
                lines.append(line)
        return lines

    @dispatcher.span
    async def achat_llm(
        self,
        query_str: str,
    ) -> ChatResponseWrapper:
        news_role_text = ""
        for line in self._get_news_role_texts():
            news_role_text += line

        return ChatResponseWrapper(
            response=ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=news_role_text,
                ),
            )
        )

    @dispatcher.span
    async def astream_chat_llm(
        self,
        query_str: str,
    ) -> ChatResponseWrapper:
        span_id = active_span_id.get()
        logger.info(f"Chat news only llm with query {query_str}")
        news_role_text = ""
        lines = self._get_news_role_texts()
        for line in lines:
            news_role_text += line
        # store role text in span output
        dispatcher.event(
            QueryEndEvent(
                response=Response(response=news_role_text, source_nodes=[]),
                query="",
                span_id=span_id,
            )
        )

        async def gen() -> ChatResponseAsyncGen:
            for line in lines:
                yield ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=line,
                    ),
                    delta=line,
                )

        return ChatResponseWrapper(response=gen())

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MiaobiNewsTool"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            num_output=self.config.top_news_count,
            is_chat_model=True,
            model_name=self.config.chat_news_model_id,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError
