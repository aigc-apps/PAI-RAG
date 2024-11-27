from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_index_manager import index_manager
from pai_rag.core.rag_module import (
    resolve_agent,
    resolve_chat_store,
    resolve_data_analysis_tool,
    resolve_data_loader,
    resolve_intent_router,
    resolve_query_engine,
    resolve_query_transform,
    resolve_searcher,
)
from pai_rag.integrations.router.pai.pai_router import Intents
from pai_rag.integrations.synthesizer.pai_synthesizer import PaiQueryBundle
from pai_rag.app.api.models import (
    RagQuery,
    RetrievalQuery,
    RagResponse,
    ContextDoc,
    RetrievalResponse,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import (
    ImageNode,
)
import json
from loguru import logger
import os
from enum import Enum
from uuid import uuid4

DEFAULT_EMPTY_RESPONSE_GEN = "Empty Response"
DEFAULT_RAG_INDEX_FILE = "localdata/default_rag_indexes.json"


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
    async for token in response.async_response_gen():
        if token and token != DEFAULT_EMPTY_RESPONSE_GEN:
            chunk = {"delta": token, "is_finished": False}
            content += token
            yield _event_chunk_wrapper(
                json.dumps(chunk, ensure_ascii=False), sse_version
            )

    if chat_store:
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


class RagApplication:
    def __init__(self, config: RagConfig):
        self.name = "RagApplication"
        self.config = config
        index_manager.add_default_index(self.config)

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
            intent = await intent_router.aselect(str_or_query_bundle=new_question)
            logger.info(f"[IntentDetection] Routing query to {intent}.")
            if intent == Intents.TOOL:
                return await self.aquery_agent(query, sse_version=sse_version)
            elif intent == Intents.WEBSEARCH:
                chat_type = RagChatType.WEB
            elif intent == Intents.NL2SQL:
                return await self.aquery_analysis(query)
            elif intent != Intents.RAG:
                return ValueError(f"Invalid intent {intent}")

        query_bundle = PaiQueryBundle(query_str=new_question, stream=query.stream)
        chat_store.add_message(
            session_id, ChatMessage(role=MessageRole.USER, content=query.question)
        )
        if chat_type == RagChatType.RAG:
            query_engine = resolve_query_engine(session_config)
            response = await query_engine.aquery(query_bundle)
        elif chat_type == RagChatType.WEB:
            search_engine = resolve_searcher(session_config)
            if not search_engine:
                raise ValueError("AI search not enabled. Please add search API key.")
            response = await search_engine.aquery(query_bundle)
        elif chat_type == RagChatType.LLM:
            query_engine = resolve_query_engine(session_config)
            query_bundle.no_retrieval = True
            response = await query_engine.asynthesize(query_bundle, nodes=[])

        node_results = response.source_nodes
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
            "new_query": new_question,
        }

        if not query.stream:
            chat_store.add_message(
                session_id,
                ChatMessage(role=MessageRole.ASSISTANT, content=response.response),
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

    async def aquery_analysis(
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

        analysis_tool = resolve_data_analysis_tool(self.config)
        if not analysis_tool:
            raise ValueError("Data Analysis not enabled. Please specify analysis type.")

        if not query.stream:
            response = await analysis_tool.aquery(query.question)
        else:
            response = await analysis_tool.astream_query(query.question)

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
            return RagResponse(answer=response.response, **result_info)
        else:
            return event_generator_async(response=response, sse_version=sse_version)

    def sql_query(self, input_list: list, sse_version: SseVersion = SseVersion.V0):
        # session_id = query.session_id or uuid_generator()
        # logger.debug(f"Get session ID: {session_id}.")

        analysis_tool = resolve_data_analysis_tool(self.config)
        if not analysis_tool:
            raise ValueError("Data Analysis not enabled. Please specify analysis type.")

        result = analysis_tool.sql_query(input_list)

        return result
