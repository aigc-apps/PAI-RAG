from pai_rag.modules.module_registry import module_registry
from pai_rag.app.api.models import (
    RagQuery,
    LlmQuery,
    RetrievalQuery,
    RagResponse,
    LlmResponse,
    ContextDoc,
    RetrievalResponse,
)
from llama_index.core.schema import QueryBundle
import json
import logging
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

    async def aload_knowledge(
        self,
        input_files,
        filter_pattern=None,
        faiss_path=None,
        enable_qa_extraction=False,
    ):
        sessioned_config = self.config
        if faiss_path:
            sessioned_config = self.config.copy()
            sessioned_config.index.update({"persist_path": faiss_path})
            self.logger.info(
                f"Update rag_application config with faiss_persist_path: {faiss_path}"
            )

        data_loader = module_registry.get_module_with_config(
            "DataLoaderModule", sessioned_config
        )
        await data_loader.aload(input_files, filter_pattern, enable_qa_extraction)

    async def aquery_retrieval(self, query: RetrievalQuery) -> RetrievalResponse:
        if not query.question:
            return RetrievalResponse(docs=[])

        sessioned_config = self.config
        if query.vector_db and query.vector_db.faiss_path:
            sessioned_config = self.config.copy()
            sessioned_config.index.update({"persist_path": query.vector_db.faiss_path})

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

    async def aquery(self, query: RagQuery):
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
            sessioned_config = self.config.copy()
            sessioned_config.index.update({"persist_path": query.vector_db.faiss_path})

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

    async def aquery_llm(self, query: LlmQuery):
        """Query answer from LLM response asynchronously.

        Generate answer from LLM's or LLM Chat Engine's achat interface.

        Args:
            query: LlmQuery

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
            return event_generator_async(response=response)

    async def aquery_agent(self, query: LlmQuery) -> LlmResponse:
        """Query answer from RAG App via web search asynchronously.

        Generate answer from agent's achat interface.

        Args:
            query: LlmQuery

        Returns:
            LlmResponse
        """
        if not query.question:
            return LlmResponse(answer="Empty query. Please input your question.")

        agent = module_registry.get_module_with_config("AgentModule", self.config)
        response = await agent.achat(query.question)
        return LlmResponse(answer=response.response)

    async def aload_evaluation_qa_dataset(self, overwrite: bool = False):
        vector_store_type = (
            self.config.get("index").get("vector_store").get("type", None)
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
            self.config.get("index").get("vector_store").get("type", None)
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
