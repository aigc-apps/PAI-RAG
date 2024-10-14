from pai_rag.integrations.query_transform.pai_query_transform import (
    PaiCondenseQueryTransform,
)
from pai_rag.integrations.synthesizer.pai_synthesizer import PaiQueryBundle
from pai_rag.modules.module_registry import module_registry
from pai_rag.app.api.models import (
    RagQuery,
    RetrievalQuery,
    RagResponse,
    LlmResponse,
    ContextDoc,
    RetrievalResponse,
)
from llama_index.core import Settings
from llama_index.core.schema import QueryBundle
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import (
    ImageNode,
)
import json
import logging
import os
import copy
from enum import Enum
from uuid import uuid4

DEFAULT_EMPTY_RESPONSE_GEN = "Empty Response"


def uuid_generator() -> str:
    return uuid4().hex


class RagChatType(str, Enum):
    LLM = "llm"
    RAG = "rag"
    WEB = "web"


async def event_generator_async(
    response, extra_info=None, chat_store=None, session_id=None
):
    content = ""
    async for token in response.async_response_gen():
        if token and token != DEFAULT_EMPTY_RESPONSE_GEN:
            chunk = {"delta": token, "is_finished": False}
            content += token
            yield json.dumps(chunk, ensure_ascii=False) + "\n"

    if chat_store:
        chat_store.add_message(
            session_id, ChatMessage(role=MessageRole.ASSISTANT, content=content)
        )

    if extra_info:
        # 返回
        last_chunk = {"delta": "", "is_finished": True, **extra_info}
    else:
        last_chunk = {"delta": "", "is_finished": True}

    yield json.dumps(last_chunk, default=lambda x: x.dict(), ensure_ascii=False)


class RagApplication:
    def __init__(self):
        self.name = "RagApplication"
        self.logger = logging.getLogger(__name__)

    def initialize(self, config):
        self.config = config
        module_registry.init_modules(self.config)
        self.chat_store: BaseChatStore = module_registry.get_module_with_config(
            "ChatStoreModule", config
        )
        self.condense_query_transform = PaiCondenseQueryTransform(
            chat_store=self.chat_store
        )
        self._llm = Settings.llm
        self.logger.info("RagApplication initialized successfully.")

    def reload(self, config):
        self.initialize(config)
        self.logger.info("RagApplication reloaded successfully.")

    def load_knowledge(
        self,
        input_files,
        filter_pattern=None,
        faiss_path=None,
        enable_qa_extraction=False,
        from_oss=False,
        oss_path=None,
        enable_raptor=False,
    ):
        sessioned_config = copy.copy(self.config)
        if faiss_path:
            sessioned_config.rag.index.update({"persist_path": faiss_path})
            self.logger.info(
                f"Update rag_application config with faiss_persist_path: {faiss_path}"
            )

        data_loader = module_registry.get_module_with_config(
            "DataLoaderModule", sessioned_config
        )
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

        sessioned_config = self.config
        if query.vector_db and query.vector_db.faiss_path:
            sessioned_config = copy.copy(self.config)
            sessioned_config.rag.index.update(
                {"persist_path": query.vector_db.faiss_path}
            )

        query_bundle = QueryBundle(query.question)

        query_engine = module_registry.get_module_with_config(
            "QueryEngineModule", sessioned_config
        )
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

    async def aquery(self, query: RagQuery, chat_type: RagChatType = RagChatType.RAG):
        session_id = query.session_id or uuid_generator()
        self.logger.debug(f"Get session ID: {session_id}.")
        if not query.question:
            return RagResponse(
                answer="Empty query. Please input your question.", session_id=session_id
            )

        sessioned_config = self.config
        if query.vector_db and query.vector_db.faiss_path:
            sessioned_config = copy.copy(self.config)
            sessioned_config.rag.index.update(
                {"persist_path": query.vector_db.faiss_path}
            )
        # Condense question
        new_query_bundle = await self.condense_query_transform.arun(
            query_bundle_or_str=query.question,
            session_id=session_id,
            chat_history=query.chat_history,
        )
        new_question = new_query_bundle.query_str
        self.logger.info(f"Querying with question '{new_question}'.")

        if query.with_intent:
            intent_detector = module_registry.get_module_with_config(
                "IntentDetectionModule", self.config
            )
            intent = await intent_detector.aselect(
                intent_detector._choices, query=query.question
            )
            self.logger.info(f"[IntentDetection] Routing query to {intent.intent}.")
            if intent.intent == "agent":
                return await self.aquery_agent(query)
            elif intent.intent != "retrieval":
                return ValueError(f"Invalid intent {intent.intent}")

        query_bundle = PaiQueryBundle(query_str=new_question, stream=query.stream)
        self.chat_store.add_message(
            session_id, ChatMessage(role=MessageRole.USER, content=query.question)
        )
        if chat_type == RagChatType.RAG:
            query_engine = module_registry.get_module_with_config(
                "QueryEngineModule", self.config
            )
            response = await query_engine.aquery(query_bundle)
        elif chat_type == RagChatType.WEB:
            search_engine = module_registry.get_module_with_config(
                "SearchModule", sessioned_config
            )
            if not search_engine:
                raise ValueError("AI search not enabled. Please add search API key.")
            response = await search_engine.aquery(query_bundle)
        elif chat_type == RagChatType.LLM:
            query_engine = module_registry.get_module_with_config(
                "QueryEngineModule", self.config
            )
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
            self.chat_store.add_message(
                session_id,
                ChatMessage(role=MessageRole.ASSISTANT, content=response.response),
            )
            return RagResponse(answer=response.response, **result_info)
        else:
            return event_generator_async(
                response=response,
                extra_info=result_info,
                chat_store=self.chat_store,
                session_id=session_id,
            )

    async def aquery_agent(self, query: RagQuery) -> RagResponse:
        """Query answer from RAG App via web search asynchronously.

        Generate answer from agent's achat interface.

        Args:
            query: RagQuery

        Returns:
            LlmResponse
        """
        if not query.question:
            return LlmResponse(answer="Empty query. Please input your question.")

        agent = module_registry.get_module_with_config("AgentModule", self.config)
        response = await agent.achat(query.question)
        return RagResponse(answer=response.response)

    # async def aquery(self, query: RagQuery):
    #    if query.with_intent:
    #        return await self.aquery_with_intent(query)
    #    else:
    #        return await self.aquery_rag(query)

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

    async def aquery_analysis(self, query: RagQuery):
        """Query answer from RAG App asynchronously.

        Generate answer from Data Analysis interface.

        Args:
            query: RagQuery

        Returns:
            RagResponse
        """
        session_id = query.session_id or uuid_generator()
        self.logger.debug(f"Get session ID: {session_id}.")
        if not query.question:
            return RagResponse(
                answer="Empty query. Please input your question.", session_id=session_id
            )

        sessioned_config = self.config

        analyst = module_registry.get_module_with_config(
            "DataAnalysisModule", sessioned_config
        )
        if not analyst:
            raise ValueError("Data Analysis not enabled. Please specify analysis type.")

        if not query.stream:
            response = await analyst.aquery(query.question, streaming=query.stream)
        else:
            response = await analyst.astream_query(query.question)

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
            return event_generator_async(response=response, extra_info=result_info)
