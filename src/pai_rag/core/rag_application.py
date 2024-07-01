from pai_rag.modules.module_registry import module_registry
from pai_rag.evaluations.batch_evaluator import BatchEvaluator
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

import logging
from uuid import uuid4


def uuid_generator() -> str:
    return uuid4().hex


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

    # TODO: 大量文件上传实现异步添加
    async def aload_knowledge(self, file_dir, enable_qa_extraction=False):
        data_loader = module_registry.get_module_with_config(
            "DataLoaderModule", self.config
        )
        await data_loader.aload(file_dir, enable_qa_extraction)

    def load_knowledge(self, file_dir, faiss_path=None, enable_qa_extraction=False):
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
        data_loader.load(file_dir, enable_qa_extraction)

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

    async def aquery(self, query: RagQuery) -> RagResponse:
        """Query answer from RAG App asynchronously.

        Generate answer from Query Engine's or Chat Engine's achat interface.

        Args:
            query: RagQuery

        Returns:
            RagResponse
        """
        session_id = query.session_id or uuid_generator()
        self.logger.info(f"Get session ID: {session_id}.")
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
        response = await query_chat_engine.achat(query.question)
        node_results = response.sources[0].raw_output.source_nodes
        reference_docs = [
            ContextDoc(
                text=score_node.node.get_content(),
                metadata=score_node.node.metadata,
                score=score_node.score,
            )
            for score_node in node_results
        ]
        new_query = response.sources[0].raw_input["query"]
        return RagResponse(
            answer=response.response,
            session_id=session_id,
            docs=reference_docs,
            new_query=new_query,
        )

    async def aquery_llm(self, query: LlmQuery) -> LlmResponse:
        """Query answer from LLM response asynchronously.

        Generate answer from LLM's or LLM Chat Engine's achat interface.

        Args:
            query: LlmQuery

        Returns:
            LlmResponse
        """
        session_id = query.session_id or uuid_generator()
        self.logger.info(f"Get session ID: {session_id}.")

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
        response = await llm_chat_engine.achat(query.question)

        return LlmResponse(answer=response.response, session_id=session_id)

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

    async def batch_evaluate_retrieval_and_response(self, type):
        retriever = module_registry.get_module_with_config(
            "RetrieverModule", self.config
        )
        query_engine = module_registry.get_module_with_config(
            "QueryEngineModule", self.config
        )
        batch_eval = BatchEvaluator(self.config, retriever, query_engine)
        df, eval_res_avg = await batch_eval.batch_retrieval_response_aevaluation(
            type=type, workers=2, save_to_file=True
        )

        return df, eval_res_avg
