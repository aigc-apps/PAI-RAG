import re
import time
from typing import AsyncGenerator, List

from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_module import (
    resolve_agent,
    resolve_chat_llm,
    resolve_llm_guardrail,
    resolve_openai_query_transform,
    resolve_query_engine,
    resolve_searcher,
    resolve_news_tool,
    resolve_data_analysis_query,
    resolve_vector_index,
)
from pai_rag.core.utils.chat_utils import (
    SseVersion,
    chat_id_generator,
    make_completion_chunk_response,
    make_completion_response,
    make_legacy_sse_chunk_async,
    make_legacy_response,
    response_gen_from_text,
    response_from_text,
)

from pai_rag.integrations.chat_store.pai.pai_chat_store import PaiChatStore
from pai_rag.integrations.query_transform.pai_query_transform import (
    messages_to_history_str,
)
from pai_rag.knowledgebase.rag_knowledgebase import KnowledgeBase, knowledgebase_manager
from pai_rag.app.api.models import (
    ChatCompletionRequest,
    ChatIntentType,
    ChatResponseWrapper,
    ChatToolType,
    ContextDoc,
    RagResponse,
)
from pai_rag.app.api.models import PaiQueryBundle
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ChatResponse,
)
from llama_index.core.schema import ImageNode

from openai.types.completion_usage import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
)
from llama_index.core.chat_engine.types import (
    StreamingAgentChatResponse,
    AgentChatResponse,
)

from loguru import logger


DEFAULT_GUARDRAIL_RESPONSE = "抱歉，无法处理这个请求。"
DEFAULT_EMPTY_RESPONSE = "看起来你发了一条空白消息，有什么能帮到你的吗？"
DEFAULT_ERROR_RESPONSE = "抱歉，系统出错，暂时无法处理这个请求。"


def message_is_empty(messages: List[ChatMessage]):
    if len(messages) == 0 or messages[-1].content is None or messages[-1].content == "":
        return True

    return False


def remove_think_from_messages(messages: List[ChatMessage]):
    new_messages = []
    for message in messages:
        if message.content is not None:
            message.content = re.sub(
                r"<think>.*?</think>\n*",
                "",
                message.content,
                flags=re.DOTALL,
            )
        new_messages.append(message)
    return new_messages


def parse_system_prompt(messages: List[ChatMessage]):
    if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM:
        system_prompt = messages[0].content
        return system_prompt, messages[1:]

    return None, messages


class ChatFlow:
    async def _recognize_intent(
        self,
        chat_request: ChatCompletionRequest,
        config: RagConfig,
    ) -> PaiQueryBundle:
        # 默认RAG
        potential_intents = [ChatToolType.CHAT_LLM]

        if config.system.default_web_search or chat_request.search_web:
            potential_intents.append(ChatToolType.SEARCH_WEB)
            # 打开Web search的时候有可能会同时使用新闻
            if chat_request.chat_news:
                potential_intents.append(ChatToolType.CHAT_NEWS)
        elif chat_request.chat_knowledgebase:
            potential_intents.append(ChatToolType.CHAT_KNOWLEDGEBASE)
        elif chat_request.chat_agent:
            potential_intents.append(ChatToolType.CHAT_AGENT)
        elif chat_request.chat_db:
            potential_intents.append(ChatToolType.CHAT_DB)
        elif chat_request.chat_llm:
            pass
        elif chat_request.chat_news:
            potential_intents.append(ChatToolType.CHAT_NEWS)
        else:
            potential_intents.append(ChatToolType.CHAT_KNOWLEDGEBASE)

        llm_kwargs = {}
        if chat_request.temperature is not None:
            llm_kwargs["temperature"] = chat_request.temperature
        if chat_request.max_tokens is not None:
            llm_kwargs["max_tokens"] = chat_request.max_tokens

        query_transform = resolve_openai_query_transform(config)
        if query_transform is not None and len(potential_intents) > 1:
            query_bundle = await query_transform.arun(
                chat_messages=chat_request.messages,
                potential_intents=potential_intents,
            )
            query_bundle.llm_kwargs = llm_kwargs
            query_bundle.stream = chat_request.stream
            query_bundle.model = chat_request.model
            return query_bundle
        else:
            logger.info("No query transform found, using default intent.")
            return PaiQueryBundle(
                query_str=chat_request.messages[-1].content,
                messages=chat_request.messages,
                intent=potential_intents[-1].value,
                stream=chat_request.stream,
                model=chat_request.model,
                chat_messages_str=messages_to_history_str(chat_request.messages[-7:-1]),
                llm_kwargs=llm_kwargs,
            )

    async def astream_chat(
        self,
        chat_request: ChatCompletionRequest,
        config: RagConfig,
    ) -> AsyncGenerator[str, None]:
        start_time = time.time()
        chat_id = chat_id_generator()
        response_wrapper = await self._achat_internal(
            chat_id=chat_id,
            chat_request=chat_request,
            config=config,
            start_time=start_time,
        )
        token_usage = CompletionUsage(
            completion_tokens=response_wrapper.additional_kwargs.get(
                "completion_tokens", 0
            ),
            prompt_tokens=response_wrapper.additional_kwargs.get("prompt_tokens", 0),
            total_tokens=response_wrapper.additional_kwargs.get("total_tokens", 0),
        )
        return make_completion_chunk_response(
            chat_id=chat_id,
            model=chat_request.model,
            response_wrapper=response_wrapper,
            base_token_usage=token_usage,
            start_time=start_time,
            return_reference=chat_request.return_reference,
        )

    async def achat(
        self,
        chat_request: ChatCompletionRequest,
        config: RagConfig,
    ) -> ChatCompletion:
        start_time = time.time()
        chat_id = chat_id_generator()
        response_wrapper = await self._achat_internal(
            chat_id=chat_id,
            chat_request=chat_request,
            config=config,
            start_time=start_time,
        )
        token_usage = CompletionUsage(
            completion_tokens=response_wrapper.additional_kwargs.get(
                "completion_tokens", 0
            ),
            prompt_tokens=response_wrapper.additional_kwargs.get("prompt_tokens", 0),
            total_tokens=response_wrapper.additional_kwargs.get("total_tokens", 0),
        )
        return make_completion_response(
            chat_id=chat_id,
            model=chat_request.model,
            response_wrapper=response_wrapper,
            base_token_usage=token_usage,
            return_reference=chat_request.return_reference,
        )

    async def aquery(
        self,
        session_id: str,
        chat_request: ChatCompletionRequest,
        config: RagConfig,
        chat_store: PaiChatStore,
        sse_version: SseVersion = SseVersion.V0,
    ) -> RagResponse:
        start_time = time.time()
        response_wrapper = await self._achat_internal(
            chat_id=session_id,
            chat_request=chat_request,
            config=config,
            start_time=start_time,
        )
        docs = []
        if chat_request.return_reference:
            docs = [
                ContextDoc(
                    text=score_node.node.text,
                    metadata=score_node.node.metadata,
                    score=score_node.score,
                    image_url=score_node.node.image_url,
                )
                if isinstance(score_node.node, ImageNode)
                else ContextDoc(
                    text=score_node.node.text,
                    metadata=score_node.node.metadata,
                    score=score_node.score,
                )
                for score_node in response_wrapper.source_nodes
            ]

        if chat_request.stream:
            return make_legacy_sse_chunk_async(
                response_wrapper=response_wrapper,
                history_messages=chat_request.messages[:-1],
                docs=docs,
                chat_store=chat_store,
                session_id=session_id,
                sse_version=sse_version,
            )
        else:
            return make_legacy_response(
                response_wrapper=response_wrapper,
                history_messages=chat_request.messages[:-1],
                docs=docs,
                chat_store=chat_store,
                session_id=session_id,
            )

    async def _achat_internal(
        self,
        chat_id: str,
        chat_request: ChatCompletionRequest,
        config: RagConfig,
        start_time: float = 0.0,
    ) -> ChatResponseWrapper:
        system_prompt, messages = parse_system_prompt(chat_request.messages)
        if message_is_empty(messages):
            if chat_request.stream:
                return response_gen_from_text(DEFAULT_EMPTY_RESPONSE)
            return response_from_text(DEFAULT_EMPTY_RESPONSE)

        chat_request.messages = remove_think_from_messages(messages)

        # 意图识别
        query_bundle = await self._recognize_intent(chat_request, config)
        query_bundle.system_role = system_prompt
        logger.info(
            f"[{chat_id}] Intent recognized: {query_bundle.intent}, query: {query_bundle.query_str}, elapsed time: {time.time() - start_time}s."
        )
        logger.info(
            f"[{chat_id}] Intent recognizede with {query_bundle.prompt_tokens} prompt tokens, {query_bundle.completion_tokens} completion tokens, {query_bundle.total_tokens} total tokens."
        )

        # 安全护栏
        guardrail = resolve_llm_guardrail(config)
        if guardrail is not None:
            check_result = await guardrail.acheck(text=query_bundle.query_str)
            if check_result.reject:
                logger.info(f"Guadrail check failed: {query_bundle.query_str}.")
                if chat_request.stream:
                    return response_gen_from_text(check_result.advice)
                else:
                    return response_from_text(check_result.advice)

            logger.info(f"Guadrail check passed: {query_bundle.query_str}.")

        # 意图分发
        if query_bundle.intent == ChatIntentType.CHAT_LLM:
            response_wrapper = await self.achat_llm(query_bundle, config=config)
        elif query_bundle.intent == ChatIntentType.CHAT_NEWS:
            response_wrapper = await self.achat_news(query_bundle, config=config)
        elif query_bundle.intent == ChatIntentType.LIST_NEWS:
            response_wrapper = await self.alist_news(query_bundle, config=config)
        elif query_bundle.intent == ChatIntentType.CHAT_AGENT:
            response_wrapper = await self.achat_agent(query_bundle, config=config)
        elif query_bundle.intent == ChatIntentType.SEARCH_WEB:
            response_wrapper = await self.achat_web(query_bundle, config=config)
        elif query_bundle.intent == ChatIntentType.CHAT_DB:
            response_wrapper = await self.achat_db(query_bundle, config=config)
        elif query_bundle.intent == ChatIntentType.CHAT_KNOWLEDGEBASE:
            knowledgebase = knowledgebase_manager.get_knowledgebase(
                chat_request.index_name
            )
            response_wrapper = await self.achat_knowledgebase(
                query_bundle, config=config, knowledgebase=knowledgebase
            )
        else:
            logger.warning(f"Unknown intent: {query_bundle.intent}")
            response_wrapper = await self.achat_llm(query_bundle, config=config)

        # 计算query_rewrite的token数量
        response_wrapper.additional_kwargs[
            "completion_tokens"
        ] = query_bundle.completion_tokens
        response_wrapper.additional_kwargs["prompt_tokens"] = query_bundle.prompt_tokens
        response_wrapper.additional_kwargs["total_tokens"] = query_bundle.total_tokens
        return response_wrapper

    async def achat_db(
        self,
        query_bundle: PaiQueryBundle,
        config: RagConfig,
    ):
        data_analysis_query_engine = resolve_data_analysis_query(config)
        if not data_analysis_query_engine:
            raise ValueError(
                "DBChat config is not valid. Please check your DBChat api configuration."
            )

        return await data_analysis_query_engine.aquery(query_bundle)

    async def alist_news(
        self,
        query_bundle: PaiQueryBundle,
        config: RagConfig,
    ):
        news_tool = resolve_news_tool(config)
        if not query_bundle.stream:
            response_wrapper = await news_tool.alist_topics()
        else:
            response_wrapper = await news_tool.astream_list_topics()
        return response_wrapper

    async def achat_news(
        self,
        query_bundle: PaiQueryBundle,
        config: RagConfig,
    ):
        news_tool = resolve_news_tool(config, model_id=query_bundle.model)
        if not query_bundle.stream:
            response_wrapper = await news_tool.achat(prompt=query_bundle.query_str)
        else:
            response_wrapper = await news_tool.astream_chat(
                prompt=query_bundle.query_str
            )

        return response_wrapper

    async def achat_web(
        self,
        query_bundle: PaiQueryBundle,
        config: RagConfig,
    ):
        search_engine = resolve_searcher(config, model_id=query_bundle.model)
        if not search_engine:
            raise ValueError(
                "Web search config is not valid. Please check your search api configuration."
            )
        return await search_engine.aquery(
            query_bundle,
        )

    async def achat_knowledgebase(
        self,
        query_bundle: PaiQueryBundle,
        config: RagConfig,
        knowledgebase: KnowledgeBase,
    ) -> ChatResponseWrapper:
        vector_index = resolve_vector_index(knowledgebase)
        query_engine = resolve_query_engine(
            config, vector_index=vector_index, model_id=query_bundle.model
        )
        response = await query_engine.aquery(query_bundle)
        return response

    async def achat_agent(
        self,
        query_bundle: PaiQueryBundle,
        config: RagConfig,
    ) -> ChatResponseWrapper:
        agent = resolve_agent(config, model_id=query_bundle.model)

        if query_bundle.stream:

            async def agent_gen():
                agent_response_gen: StreamingAgentChatResponse = (
                    await agent.astream_chat(message=query_bundle.query_str)
                )
                message_content = ""
                async for token in agent_response_gen.async_response_gen():
                    message_content += token
                    yield ChatResponse(
                        delta=token,
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT, content=message_content
                        ),
                        additional_kwargs={},
                    )

            return ChatResponseWrapper(response=agent_gen())
        else:
            agent_response: AgentChatResponse = await agent.achat(
                message=query_bundle.query_str
            )
            return ChatResponseWrapper(
                response=ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=agent_response.response,
                    ),
                    additional_kwargs={},
                )
            )

    async def achat_llm(
        self,
        query_bundle: PaiQueryBundle,
        config: RagConfig,
    ) -> ChatResponseWrapper:
        llm = resolve_chat_llm(config, model_id=query_bundle.model)
        system_role = (
            query_bundle.system_role or config.synthesizer.system_role_template
        )
        messages = query_bundle.messages
        if system_role:
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_role)
            ] + query_bundle.messages

        if query_bundle.stream:
            response_gen = await llm.astream_chat(messages, **query_bundle.llm_kwargs)
            return ChatResponseWrapper(response=response_gen)
        else:
            response = await llm.achat(messages, **query_bundle.llm_kwargs)
            return ChatResponseWrapper(response=response)
