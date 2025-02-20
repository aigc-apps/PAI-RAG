from pai_rag.app.api.models import ChatCompletionRequest
from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_index_manager import index_manager
from pai_rag.core.rag_module import (
    resolve_agent,
    resolve_chat_store,
    resolve_data_analysis_loader,
    resolve_data_analysis_query,
    resolve_data_loader,
    resolve_intent_router,
    resolve_llm_guardrail,
    resolve_query_engine,
    resolve_query_transform,
    resolve_searcher,
    resolve_openai_query_transform,
)
from pai_rag.integrations.router.pai.pai_router import Intents
from pai_rag.app.api.models import PaiQueryBundle
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice

import openai.types.chat.chat_completion_chunk as chat_completion_chunk
from openai._exceptions import APIError
from openai.types.completion_usage import CompletionUsage
from pai_rag.app.api.models import (
    RagQuery,
    RetrievalQuery,
    RagResponse,
    ContextDoc,
    RetrievalResponse,
)
from llama_index.core.base.llms.generic_utils import messages_to_history_str
from llama_index.core.schema import QueryBundle
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import (
    ImageNode,
)
import json
import os
from loguru import logger
from enum import Enum
from uuid import uuid4
from llama_index.core import Settings
import time
import re

DEFAULT_RAG_INDEX_FILE = "localdata/default_rag_indexes.json"
DEFAULT_GUARDRAIL_RESPONSE = "抱歉，无法处理这个请求。"


def uuid_generator() -> str:
    return uuid4().hex


class RagChatType(str, Enum):
    LLM = "llm"
    RAG = "rag"
    WEB = "web"


class SseVersion(int, Enum):
    V0 = 0  # Backward compatibility
    V1 = 1  # New V1 version


def _event_chunk_wrapper(chunk_content, sse_version: SseVersion = SseVersion.V0):
    if sse_version == sse_version.V1:
        return f"data: {chunk_content}\n\n"
    else:
        return f"{chunk_content}\n"


async def event_generator_async(
    response,
    extra_info=None,
    chat_store=None,
    session_id=None,
    sse_version: SseVersion = SseVersion.V0,
):
    content = ""
    if isinstance(response, str):
        content = response
        chunk = {"delta": content, "is_finished": False}
        yield _event_chunk_wrapper(json.dumps(chunk, ensure_ascii=False), sse_version)

    else:
        async for token in response.async_response_gen():
            if token:
                chunk = {"delta": token, "is_finished": False}
                content += token
                yield _event_chunk_wrapper(
                    json.dumps(chunk, ensure_ascii=False), sse_version
                )

    if chat_store:
        content = re.sub(r"<think>.*?</think>\n*", "", content, flags=re.DOTALL)
        chat_store.add_message(
            session_id, ChatMessage(role=MessageRole.ASSISTANT, content=content)
        )

    if extra_info:
        # 返回
        last_chunk = {"delta": "", "is_finished": True, **extra_info}
    else:
        last_chunk = {"delta": "", "is_finished": True}

    last_chunk_data = json.dumps(
        last_chunk, default=lambda x: x.dict(), ensure_ascii=False
    )
    yield _event_chunk_wrapper(last_chunk_data, sse_version)


def _make_chat_completion_response(session_id, response, return_reference=False):
    logger.info(f"Finished response: {response.response}")
    citations = []
    citation_details = []
    if return_reference:
        for score_node in response.source_nodes:
            if isinstance(score_node.node, ImageNode):
                if score_node.image_url is not None:
                    citations.append(score_node.image_url)
                    citation_details.append(
                        {
                            "name": "Image",
                            "text": None,
                            "url": score_node.image_url,
                            "score": score_node.score,
                        }
                    )
            else:
                url = score_node.node.metadata.get(
                    "file_url"
                ) or score_node.node.metadata.get("file_path")
                citations.append(url)
                citation_details.append(
                    {
                        "name": score_node.node.metadata.get("file_name"),
                        "text": score_node.node.text,
                        "url": url,
                        "score": score_node.score,
                    }
                )

    return ChatCompletion(
        id=session_id,
        created=int(time.time()),
        model=Settings.llm.metadata.model_name,
        citations=citations,
        citation_details=citation_details,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role=MessageRole.ASSISTANT.value,
                    content=response.response,
                ),
                finish_reason="stop",
            )
        ],
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
        ),
    )


def _make_chat_completion_response_with_text(session_id, text):
    logger.info(f"Finished response: {text}")
    return ChatCompletion(
        id=session_id,
        created=int(time.time()),
        model=Settings.llm.metadata.model_name,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role=MessageRole.ASSISTANT.value,
                    content=text,
                ),
                finish_reason="stop",
            )
        ],
        citations=[],
        citation_details=[],
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
        ),
    )


async def _make_chat_completion_chunk_response(
    session_id, response, return_reference=False
):
    i = 0
    full_content = ""
    created_ts = int(time.time())
    citations = []
    citation_details = []
    if return_reference:
        for score_node in response.source_nodes:
            if isinstance(score_node.node, ImageNode):
                if score_node.image_url is not None:
                    citations.append(score_node.image_url)
                    citation_details.append(
                        {
                            "name": "Image",
                            "text": None,
                            "url": score_node.image_url,
                            "score": score_node.score,
                        }
                    )
            else:
                url = score_node.node.metadata.get(
                    "file_url"
                ) or score_node.node.metadata.get("file_path")
                citations.append(url)
                citation_details.append(
                    {
                        "name": score_node.node.metadata.get("file_name"),
                        "url": url,
                        "text": score_node.node.text,
                        "score": score_node.score,
                    }
                )

    model_name = Settings.llm.metadata.model_name
    try:
        async for token in response.async_response_gen():
            if token:
                full_content += token
                chunk = ChatCompletionChunk(
                    id=session_id,
                    created=created_ts,
                    model=model_name,
                    citations=citations,
                    citation_details=citation_details,
                    choices=[
                        chat_completion_chunk.Choice(
                            index=i,
                            delta=chat_completion_chunk.ChoiceDelta(
                                role=MessageRole.ASSISTANT.value,
                                content=token,
                            ),
                            finish_reason=None,
                        )
                    ],
                    object="chat.completion.chunk",
                )
                i += 1
                yield f"data: {json.dumps(chunk.model_dump(mode='json'), ensure_ascii=False)}\n\n"

        last_chunk = ChatCompletionChunk(
            id=session_id,
            created=created_ts,
            model=model_name,
            citations=citations,
            citation_details=citation_details,
            choices=[
                chat_completion_chunk.Choice(
                    index=i,
                    delta=chat_completion_chunk.ChoiceDelta(
                        role=MessageRole.ASSISTANT.value,
                        content="",
                    ),
                    finish_reason="stop",
                )
            ],
            object="chat.completion.chunk",
        )
        yield f"data: {json.dumps(last_chunk.model_dump(mode='json'), ensure_ascii=False)}\n\n"

    except APIError as exception:
        logger.info(f"Streaming failed: {exception}")
        chunk = ChatCompletionChunk(
            id=session_id,
            created=created_ts,
            model=model_name,
            citations=citations,
            citation_details=citation_details,
            choices=[
                chat_completion_chunk.Choice(
                    index=i,
                    delta=chat_completion_chunk.ChoiceDelta(
                        role=MessageRole.ASSISTANT.value,
                        content=exception.message,
                    ),
                    finish_reason="stop",
                )
            ],
            object="chat.completion.chunk",
        )

        yield f"data: {json.dumps(chunk.model_dump(mode='json'), ensure_ascii=False)}\n\n"
    except Exception as exception:
        logger.info(f"Streaming failed: {exception}")
        raise exception

    logger.info(f"Finished streaming: {full_content}")


async def _make_chat_completion_chunk_response_with_text(session_id, text):
    i = 0
    created_ts = int(time.time())
    model_name = Settings.llm.metadata.model_name
    chunk = ChatCompletionChunk(
        id=session_id,
        created=created_ts,
        model=model_name,
        citations=[],
        citation_details=[],
        choices=[
            chat_completion_chunk.Choice(
                index=i,
                delta=chat_completion_chunk.ChoiceDelta(
                    role=MessageRole.ASSISTANT.value,
                    content=text,
                ),
                finish_reason="stop",
            )
        ],
        object="chat.completion.chunk",
    )
    i += 1

    logger.info(f"Finished streaming: {text}")
    yield f"data: {json.dumps(chunk.model_dump(mode='json'), ensure_ascii=False)}\n\n"


class RagApplication:
    def __init__(self, config: RagConfig):
        self.name = "RagApplication"
        self.config = config
        index_manager.add_default_index(self.config)
        _ = resolve_query_engine(self.config)

    def refresh(self, config: RagConfig):
        self.config = config
        index_manager.add_default_index(self.config)

    def load_knowledge(
        self,
        input_files,
        filter_pattern=None,
        index_name=None,
        from_oss=False,
        oss_path=None,
        enable_raptor=False,
        enable_multimodal=False,
    ):
        logger.info(
            f"""Loading data:
            input_files: {input_files}
            index_name: {index_name}
            enable_multimodal: {enable_multimodal}
            enable_raptor: {enable_raptor}"""
        )

        session_config = self.config.model_copy()
        index_entry = index_manager.get_index_by_name(index_name)
        session_config.embedding = index_entry.embedding_config
        session_config.index.vector_store = index_entry.vector_store_config
        session_config.node_parser.enable_multimodal = enable_multimodal

        data_loader = resolve_data_loader(session_config)
        data_loader.load_data(
            file_path_or_directory=input_files,
            filter_pattern=filter_pattern,
            from_oss=from_oss,
            oss_path=oss_path,
            enable_raptor=enable_raptor,
        )

    async def aretrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        if not query.question:
            return RetrievalResponse(docs=[])

        query_bundle = QueryBundle(query.question)
        session_config = self.config.model_copy()
        index_entry = index_manager.get_index_by_name(query.index_name)
        session_config.embedding = index_entry.embedding_config
        session_config.index.vector_store = index_entry.vector_store_config
        query_engine = resolve_query_engine(session_config)
        node_results = await query_engine.aretrieve(query_bundle)

        docs = [
            ContextDoc(
                text=score_node.node.get_content(),
                metadata=score_node.node.metadata,
                score=score_node.score,
                image_url=score_node.node.image_url,
            )
            if isinstance(score_node.node, ImageNode)
            else ContextDoc(
                text=score_node.node.get_content(),
                metadata=score_node.node.metadata,
                score=score_node.score,
            )
            for score_node in node_results
        ]

        return RetrievalResponse(docs=docs)

    async def achat(
        self,
        chat_request: ChatCompletionRequest,
    ):
        session_id = uuid_generator()

        if (
            len(chat_request.messages) == 0
            or chat_request.messages[-1].content is None
            or chat_request.messages[-1].content == ""
        ):
            if chat_request.stream:
                return _make_chat_completion_chunk_response_with_text(
                    session_id, "看起来你问了一个空问题，请问有什么能帮忙的吗？"
                )
            else:
                return _make_chat_completion_response_with_text(
                    session_id, "看起来你问了一个空问题，请问有什么能帮忙的吗？"
                )

        try:
            guardrail = resolve_llm_guardrail(self.config)
            passed_guardrail = False if guardrail is not None else True

            messages = chat_request.messages
            system_prompt = None
            if messages[0].role == MessageRole.SYSTEM:
                system_prompt = messages[0].content
                messages = messages[1:]

            if not passed_guardrail:
                user_messages = [
                    msg for msg in messages if msg.role == MessageRole.USER
                ]
                # 只有一条对话，直接检查
                if len(user_messages) == 1:
                    guardrail_result = await guardrail.acheck(user_messages[0].content)
                    if guardrail_result.reject:
                        if chat_request.stream:
                            return _make_chat_completion_chunk_response_with_text(
                                session_id, guardrail_result.advice
                            )
                        else:
                            return _make_chat_completion_response_with_text(
                                session_id, guardrail_result.advice
                            )
                    passed_guardrail = True

            if self.config.system.default_web_search:
                chat_request.search_web = True

            question = messages[-1].content

            openai_query_transform = resolve_openai_query_transform(self.config)
            if openai_query_transform is not None:
                new_query_bundle = await openai_query_transform.arun(
                    chat_messages=messages,
                )
            else:
                new_query_bundle = PaiQueryBundle(
                    query_str=question, need_web_search=chat_request.search_web
                )

            new_question = new_query_bundle.query_str
            if not passed_guardrail:
                # 多轮对话，用新查询检查
                guardrail_result = await guardrail.acheck(new_question)
                if guardrail_result.reject:
                    if chat_request.stream:
                        return _make_chat_completion_chunk_response_with_text(
                            session_id, guardrail_result.advice
                        )
                    else:
                        return _make_chat_completion_response_with_text(
                            session_id, guardrail_result.advice
                        )
                passed_guardrail = True

            logger.info(f"Querying with question '{new_question}'.")
            if new_question != question:
                messages[-1].content = ",".join([question, new_question])

            query_bundle = PaiQueryBundle(
                query_str=new_question,
                stream=chat_request.stream,
                citation=chat_request.citation,
                need_web_search=new_query_bundle.need_web_search,
                chat_messages_str=messages_to_history_str(messages=messages[-8:]),
            )

            if chat_request.force_no_search:
                chat_request.search_web = True
                query_bundle.need_web_search = False
            elif chat_request.force_search_web:
                chat_request.search_web = True
                query_bundle.need_web_search = True
            elif chat_request.force_search_knowledgebase:
                chat_request.search_web = False

            if chat_request.search_web:
                search_engine = resolve_searcher(self.config)
                if not search_engine:
                    raise ValueError(
                        "AI search config is not valid. Please check your search api configuration."
                    )

                response = await search_engine.aquery(
                    query_bundle,
                    system_role_str=system_prompt,
                    prompt_template_str=" " if system_prompt else None,
                )
                if chat_request.stream:
                    return _make_chat_completion_chunk_response(
                        session_id=session_id,
                        response=response,
                        return_reference=chat_request.return_reference,
                    )
                else:
                    return _make_chat_completion_response(
                        session_id=session_id,
                        response=response,
                        return_reference=chat_request.return_reference,
                    )

            session_config = self.config.model_copy()
            index_entry = index_manager.get_index_by_name(chat_request.index_name)
            session_config.embedding = index_entry.embedding_config
            session_config.index.vector_store = index_entry.vector_store_config
            query_engine = resolve_query_engine(session_config)
            response = await query_engine.aquery(
                query_bundle,
                system_role_str=system_prompt,
                prompt_template_str=" " if system_prompt else None,
            )
            if chat_request.stream:
                return _make_chat_completion_chunk_response(
                    session_id=session_id,
                    response=response,
                    return_reference=chat_request.return_reference,
                )
            else:
                return _make_chat_completion_response(
                    session_id=session_id,
                    response=response,
                    return_reference=chat_request.return_reference,
                )
        except Exception as e:
            logger.error(f"Error while processing request: {e}")
            if chat_request.stream:
                return _make_chat_completion_chunk_response_with_text(
                    session_id, "抱歉，系统错误，暂时无法处理这个请求。"
                )
            else:
                return _make_chat_completion_response_with_text(
                    session_id, "抱歉，系统错误，暂时无法处理这个请求。"
                )

    async def aquery(
        self,
        query: RagQuery,
        chat_type: RagChatType = RagChatType.RAG,
        sse_version: SseVersion = SseVersion.V0,
    ):
        session_id = query.session_id or uuid_generator()
        logger.debug(f"Get session ID: {session_id}.")
        session_config = self.config.model_copy()
        index_entry = index_manager.get_index_by_name(query.index_name)
        session_config.embedding = index_entry.embedding_config
        session_config.index.vector_store = index_entry.vector_store_config

        if not query.question:
            return RagResponse(
                answer="Empty query. Please input your question.", session_id=session_id
            )

        chat_store = resolve_chat_store(session_config)
        condense_query_transform = resolve_query_transform(session_config)

        # Condense question
        new_query_bundle = await condense_query_transform.arun(
            query_bundle_or_str=query.question,
            session_id=session_id,
            chat_history=query.chat_history,
        )
        new_question = new_query_bundle.query_str
        logger.info(f"Querying with question '{new_question}'.")

        if query.with_intent:
            intent_router = resolve_intent_router(session_config)
            intent = await intent_router.aselect(
                str_or_query_bundle=new_query_bundle.chat_messages_str
            )
            logger.info(f"[IntentDetection] Routing query to {intent}.")
            if intent == Intents.TOOL:
                return await self.aquery_agent(query, sse_version=sse_version)
            elif intent == Intents.WEBSEARCH:
                chat_type = RagChatType.WEB
            elif intent == Intents.NL2SQL:
                return await self.aquery_data_analysis(query)
            elif intent != Intents.RAG:
                return ValueError(f"Invalid intent {intent}")

        guardrail = resolve_llm_guardrail(session_config)
        # 多轮对话，用新查询检查
        if guardrail is not None:
            guardrail_result = await guardrail.acheck(new_question)
            if guardrail_result.reject:
                if query.stream:
                    return event_generator_async(
                        response=guardrail_result.advice,
                        chat_store=chat_store,
                        session_id=session_id,
                        sse_version=sse_version,
                    )

                else:
                    return RagResponse(
                        answer=guardrail_result.advice, session_id=session_id
                    )

        query_bundle = PaiQueryBundle(
            query_str=new_question,
            need_web_search=new_query_bundle.need_web_search,
            stream=query.stream,
            citation=query.citation,
            chat_messages_str=new_query_bundle.chat_messages_str,
        )
        chat_store.add_message(
            session_id, ChatMessage(role=MessageRole.USER, content=query.question)
        )
        if chat_type == RagChatType.RAG:
            query_engine = resolve_query_engine(session_config)
            response = await query_engine.aquery(
                query_bundle,
                system_role_str=query.system_role_template,
                prompt_template_str=query.custom_prompt_template,
            )
        elif chat_type == RagChatType.WEB:
            search_engine = resolve_searcher(session_config)
            if not search_engine:
                raise ValueError(
                    "AI search config is not valid. Please check your search api configuration."
                )
            response = await search_engine.aquery(
                query_bundle,
                system_role_str=query.system_role_template,
                prompt_template_str=query.custom_prompt_template,
            )
        elif chat_type == RagChatType.LLM:
            query_engine = resolve_query_engine(session_config)
            query_bundle.no_retrieval = True
            response = await query_engine.asynthesize(
                query_bundle,
                nodes=[],
                system_role_str=query.system_role_template,
                prompt_template_str=query.custom_prompt_template,
            )
        node_results = response.source_nodes
        result_info = {
            "session_id": session_id,
            "new_query": new_question,
        }

        if query.return_reference:
            reference_docs = [
                ContextDoc(
                    text=score_node.node.get_content(),
                    metadata=score_node.node.metadata,
                    score=score_node.score,
                    image_url=score_node.node.image_url,
                )
                if isinstance(score_node.node, ImageNode)
                else ContextDoc(
                    text=score_node.node.get_content(),
                    metadata=score_node.node.metadata,
                    score=score_node.score,
                )
                for score_node in node_results
            ]

            result_info["docs"] = reference_docs

        if not query.stream:
            content = re.sub(
                r"<think>.*?</think>\n*", "", response.response, flags=re.DOTALL
            )
            chat_store.add_message(
                session_id,
                ChatMessage(role=MessageRole.ASSISTANT, content=content),
            )
            return RagResponse(answer=response.response, **result_info)
        else:
            return event_generator_async(
                response=response,
                extra_info=result_info,
                chat_store=chat_store,
                session_id=session_id,
                sse_version=sse_version,
            )

    async def aquery_agent(
        self, query: RagQuery, sse_version: SseVersion = SseVersion.V0
    ) -> RagResponse:
        """Query answer from RAG App via web search asynchronously.

        Generate answer from agent's achat interface.

        Args:
            query: RagQuery

        Returns:
            RagResponse
        """
        if not query.question:
            return RagResponse(answer="Empty query. Please input your question.")

        agent = resolve_agent(self.config)
        if query.stream:
            response = await agent.astream_chat(query.question)
            return event_generator_async(response, sse_version=sse_version)
        else:
            response = await agent.achat(query.question)
            return RagResponse(answer=response.response)

    async def aload_agent_config(self, agent_cfg_path: str):
        if os.path.exists(agent_cfg_path):
            sessioned_config = self.config.as_dict().copy()
            sessioned_config["RAG"]["llm"]["function_calling_llm"][
                "source"
            ] = "DashScope"
            sessioned_config["RAG"]["llm"]["function_calling_llm"][
                "name"
            ] = "qwen2-7b-instruct"
            sessioned_config["RAG"]["agent"]["type"] = "function_calling"
            sessioned_config["RAG"]["agent"]["custom_config"][
                "agent_file_path"
            ] = agent_cfg_path
            sessioned_config["RAG"]["agent"]["intent_detection"]["type"] = "single"
            sessioned_config["RAG"]["agent"]["tool"]["type"] = "api"

            new_settings = self.config
            new_settings.update(sessioned_config)

            self.reload(new_settings)
            return "Update agent config successfully."
        else:
            return f"The agent config path {agent_cfg_path} not exists."

    async def aload_db_info(self):
        db_info_loader = resolve_data_analysis_loader(self.config)
        await db_info_loader.aload_db_info()

        return "Load database info successfully."

    async def aquery_data_analysis(
        self, query: RagQuery, sse_version: SseVersion = SseVersion.V0
    ):
        """Query answer from RAG App asynchronously.

        Generate answer from Data Analysis interface.

        Args:
            query: RagQuery

        Returns:
            RagResponse
        """
        session_id = query.session_id or uuid_generator()
        logger.debug(f"Get session ID: {session_id}.")
        if not query.question:
            return RagResponse(
                answer="Empty query. Please input your question.", session_id=session_id
            )

        analysis_query = resolve_data_analysis_query(self.config)
        if not analysis_query:
            raise ValueError("Data Analysis not enabled. Please specify analysis type.")

        if not query.stream:
            response = await analysis_query.aquery(query.question)
        else:
            response = await analysis_query.astream_query(query.question)

        node_results = response.source_nodes
        new_query = query.question

        reference_docs = [
            ContextDoc(
                text=score_node.node.get_content(),
                metadata=score_node.node.metadata,
                score=score_node.score,
                image_url=score_node.node.image_url,
            )
            if isinstance(score_node.node, ImageNode)
            else ContextDoc(
                text=score_node.node.get_content(),
                metadata=score_node.node.metadata,
                score=score_node.score,
            )
            for score_node in node_results
        ]

        result_info = {
            "session_id": session_id,
            "docs": reference_docs,
            "new_query": new_query,
        }

        if not query.stream:
            content = re.sub(
                r"<think>.*?</think>\n*", "", response.response, flags=re.DOTALL
            )
            return RagResponse(answer=content, **result_info)
        else:
            return event_generator_async(
                response=response, extra_info=result_info, sse_version=sse_version
            )
