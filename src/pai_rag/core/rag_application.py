from pai_rag.modules.module_registry import module_registry
from pai_rag.app.api.models import (
    RagQuery,
    RetrievalQuery,
    RagResponse,
    LlmResponse,
    ContextDoc,
    RetrievalResponse,
)
from llama_index.core.schema import QueryBundle
import json
import logging
import os
import copy
from uuid import uuid4

DEFAULT_EMPTY_RESPONSE_GEN = "Empty Response"


def uuid_generator() -> str:
    return uuid4().hex


async def event_generator_async(response, extra_info=None):
    content = ""
    async for token in response.async_response_gen():
        if token and token != DEFAULT_EMPTY_RESPONSE_GEN:
            chunk = {"delta": token, "is_finished": False}
            content += token
            yield json.dumps(chunk) + "\n"

    if extra_info:
        # 返回
        last_chunk = {"delta": "", "is_finished": True, **extra_info}
    else:
        last_chunk = {"delta": "", "is_finished": True}

    yield json.dumps(last_chunk, default=lambda x: x.dict())


class RagApplication:
    def __init__(self):
        self.name = "RagApplication"
        self.logger = logging.getLogger(__name__)

    def initialize(self, config):
        self.config = config
        module_registry.init_modules(self.config)
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
        enable_raptor=False,
    ):
        sessioned_config = self.config
        sessioned_config.rag.data_loader.update({"type": "Local"})
        if faiss_path:
            sessioned_config = copy.copy(self.config)
            sessioned_config.rag.index.update({"persist_path": faiss_path})
            self.logger.info(
                f"Update rag_application config with faiss_persist_path: {faiss_path}"
            )

        data_loader = module_registry.get_module_with_config(
            "DataLoaderModule", sessioned_config
        )
        data_loader.load(
            input_files, filter_pattern, enable_qa_extraction, enable_raptor
        )

    def load_knowledge_from_oss(
        self,
        filter_pattern=None,
        oss_prefix=None,
        faiss_path=None,
        enable_qa_extraction=False,
        enable_raptor=False,
    ):
        sessioned_config = copy.copy(self.config)
        sessioned_config.rag.data_loader.update({"type": "Oss"})
        sessioned_config.rag.oss_store.update({"prefix": oss_prefix})
        _ = module_registry.get_module_with_config("OssCacheModule", sessioned_config)
        self.logger.info(
            f"Update rag_application config with data_loader type: Oss and Oss Bucket prefix: {oss_prefix}"
        )
        data_loader = module_registry.get_module_with_config(
            "DataLoaderModule", sessioned_config
        )
        if faiss_path:
            sessioned_config.rag.index.update({"persist_path": faiss_path})
            self.logger.info(
                f"Update rag_application config with faiss_persist_path: {faiss_path}"
            )
        data_loader.load(
            filter_pattern=filter_pattern,
            enable_qa_extraction=enable_qa_extraction,
            enable_raptor=enable_raptor,
        )

    async def aload_knowledge_from_oss(
        self,
        filter_pattern=None,
        oss_prefix=None,
        faiss_path=None,
        enable_qa_extraction=False,
        enable_raptor=False,
    ):
        sessioned_config = copy.copy(self.config)
        sessioned_config.rag.data_loader.update({"type": "Oss"})
        sessioned_config.rag.oss_store.update({"prefix": oss_prefix})
        _ = module_registry.get_module_with_config("OssCacheModule", sessioned_config)
        self.logger.info(
            f"Update rag_application config with data_loader type: Oss and Oss Bucket prefix: {oss_prefix}"
        )
        data_loader = module_registry.get_module_with_config(
            "DataLoaderModule", sessioned_config
        )
        if faiss_path:
            sessioned_config.rag.index.update({"persist_path": faiss_path})
            self.logger.info(
                f"Update rag_application config with faiss_persist_path: {faiss_path}"
            )
        await data_loader.aload(
            filter_pattern=filter_pattern,
            enable_qa_extraction=enable_qa_extraction,
            enable_raptor=enable_raptor,
        )

    async def aquery_retrieval(self, query: RetrievalQuery) -> RetrievalResponse:
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
            )
            for score_node in node_results
        ]

        return RetrievalResponse(docs=docs)

    async def aquery_search(self, query: RagQuery):
        """Query answer from RAG App asynchronously.

        Generate answer from Query Engine's or Chat Engine's achat interface.

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

        searcher = module_registry.get_module_with_config(
            "SearchModule", sessioned_config
        )
        if not searcher:
            raise ValueError("AI search not enabled. Please add search API key.")
        if not query.stream:
            response = await searcher.aquery(query.question)
        else:
            response = await searcher.astream_query(query.question)

        node_results = response.source_nodes
        new_query = query.question

        reference_docs = [
            ContextDoc(
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

    async def aquery_rag(self, query: RagQuery):
        """Query answer from RAG App asynchronously.

        Generate answer from Query Engine's or Chat Engine's achat interface.

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

        if query.vector_db and query.vector_db.faiss_path:
            sessioned_config = copy.copy(self.config)
            sessioned_config.rag.index.update(
                {"persist_path": query.vector_db.faiss_path}
            )

        chat_engine_factory = module_registry.get_module_with_config(
            "ChatEngineFactoryModule", sessioned_config
        )
        query_chat_engine = chat_engine_factory.get_chat_engine(
            session_id, query.chat_history
        )
        if not query.stream:
            response = await query_chat_engine.achat(query.question)
        else:
            response = await query_chat_engine.astream_chat(query.question)

        node_results = response.sources[0].raw_output.source_nodes
        new_query = response.sources[0].raw_input["query"]

        reference_docs = [
            ContextDoc(
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

    async def aquery_llm(self, query: RagQuery):
        """Query answer from LLM response asynchronously.

        Generate answer from LLM's or LLM Chat Engine's achat interface.

        Args:
            query: RagQuery

        Returns:
            LlmResponse
        """
        session_id = query.session_id or uuid_generator()
        self.logger.debug(f"Get session ID: {session_id}.")

        if not query.question:
            return LlmResponse(
                answer="Empty query. Please input your question.", session_id=session_id
            )

        llm_chat_engine_factory = module_registry.get_module_with_config(
            "LlmChatEngineFactoryModule", self.config
        )
        llm_chat_engine = llm_chat_engine_factory.get_chat_engine(
            session_id, query.chat_history
        )
        if not query.stream:
            response = await llm_chat_engine.achat(query.question)
            return LlmResponse(answer=response.response, session_id=session_id)
        else:
            response = await llm_chat_engine.astream_chat(query.question)
            result_info = {"session_id": session_id}
            return event_generator_async(response=response, extra_info=result_info)

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

    async def aquery_with_intent(self, query: RagQuery):
        """Query answer from RAG App asynchronously.

        Generate answer from Query Engine's or Chat Engine's achat interface.

        Args:
            query: RagQuery

        Returns:
            RagResponse
        """
        if not query.question:
            return RagResponse(answer="Empty query. Please input your question.")
        intent_detector = module_registry.get_module_with_config(
            "IntentDetectionModule", self.config
        )
        intent = await intent_detector.aselect(
            intent_detector._choices, query=query.question
        )
        self.logger.info(f"[IntentDetection] Routing query to {intent.intent}.")
        if intent.intent == "agent":
            return await self.aquery_agent(query)
        elif intent.intent == "retrieval":
            return await self.aquery_rag(query)
        else:
            return ValueError(f"Invalid intent {intent.intent}")

    async def aquery(self, query: RagQuery):
        if query.with_intent:
            return await self.aquery_with_intent(query)
        else:
            return await self.aquery_rag(query)

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

    async def aload_evaluation_qa_dataset(self, overwrite: bool = False):
        vector_store_type = (
            self.config.rag.get("index").get("vector_store").get("type", None)
        )
        if vector_store_type == "FAISS":
            evaluation = module_registry.get_module_with_config(
                "EvaluationModule", self.config
            )
            qa_dataset = await evaluation.aload_question_answer_pairs_json(overwrite)
            return qa_dataset
        else:
            return f"Evaluation against vector store '{vector_store_type}' is not supported. Only FAISS is supported for now."

    async def aevaluate_retrieval_and_response(self, type, overwrite: bool = False):
        vector_store_type = (
            self.config.rag.get("index").get("vector_store").get("type", None)
        )
        if vector_store_type == "FAISS":
            evaluation = module_registry.get_module_with_config(
                "EvaluationModule", self.config
            )
            df, eval_res_avg = await evaluation.abatch_retrieval_response_aevaluation(
                type=type, workers=4, overwrite=overwrite
            )

            return df, eval_res_avg
        else:
            return (
                None,
                f"Evaluation against vector store '{vector_store_type}' is not supported. Only FAISS is supported for now.",
            )

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
            response = await analyst.aquery(query.question)
        else:
            response = await analyst.astream_query(query.question)

        node_results = response.source_nodes
        new_query = query.question

        reference_docs = [
            ContextDoc(
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
