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

        chat_store = module_registry.get_module_with_config(
            "ChatStoreModule", sessioned_config
        )
        chat_store.persist()
        return RagResponse(answer=response.response, session_id=session_id)

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

        chat_store = module_registry.get_module_with_config(
            "ChatStoreModule", self.config
        )
        chat_store.persist()
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
            return f"Not support for vector store —— {vector_store_type}"

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
            return None, f"Not support for vector store —— {vector_store_type}"
