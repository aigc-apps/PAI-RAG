from pai_rag.data.rag_dataloader import RagDataLoader
from pai_rag.utils.oss_cache import OssCache
from pai_rag.modules.module_registry import module_registry
from pai_rag.evaluations.batch_evaluator import BatchEvaluator
from pai_rag.app.api.models import (
    RagQuery,
    LlmQuery,
    RagResponse,
    RetrievalQuery,
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
        logging.basicConfig(level=logging.INFO)  # 将日志级别设置为INFO
        self.logger = logging.getLogger(__name__)

    def initialize(self, config):
        self.config = config

        module_registry.init_modules(self.config)
        self.index = module_registry.get_module("IndexModule")
        self.llm = module_registry.get_module("LlmModule")
        self.retriever = module_registry.get_module("RetrieverModule")
        self.chat_store = module_registry.get_module("ChatStoreModule")
        self.query_engine = module_registry.get_module("QueryEngineModule")
        self.chat_engine_factory = module_registry.get_module("ChatEngineFactoryModule")
        self.llm_chat_engine_factory = module_registry.get_module(
            "LlmChatEngineFactoryModule"
        )
        self.data_reader_factory = module_registry.get_module("DataReaderFactoryModule")
        self.agent = module_registry.get_module("AgentModule")

        oss_cache = None
        if config.get("oss_cache", None):
            oss_cache = OssCache(config.oss_cache)
        node_parser = module_registry.get_module("NodeParserModule")

        self.data_loader = RagDataLoader(
            self.data_reader_factory, node_parser, self.index, oss_cache
        )
        self.logger.info("RagApplication initialized successfully.")

    def reload(self, config):
        self.initialize(config)
        self.logger.info("RagApplication reloaded successfully.")

    # TODO: 大量文件上传实现异步添加
    def load_knowledge(self, file_dir, enable_qa_extraction=False):
        self.data_loader.load(file_dir, enable_qa_extraction)

    async def aquery_retrieval(self, query: RetrievalQuery) -> RetrievalResponse:
        if not query.question:
            return RetrievalResponse(docs=[])

        query_bundle = QueryBundle(query.question)
        node_results = await self.query_engine.aretrieve(query_bundle)

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
        self.logger.info(f"Get session ID: {session_id}.")
        if not query.question:
            return RagResponse(
                answer="Empty query. Please input your question.", session_id=session_id
            )

        query_chat_engine = self.chat_engine_factory.get_chat_engine(
            session_id, query.chat_history
        )
        if query.stream:
            response = await query_chat_engine.astream_chat(query.question)
        else:
            response = await query_chat_engine.achat(query.question)
        self.chat_store.persist()
        return RagResponse(answer=response.response, session_id=session_id)

    async def aquery_llm(self, query: LlmQuery):
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

        llm_chat_engine = self.llm_chat_engine_factory.get_chat_engine(
            session_id, query.chat_history
        )
        if query.stream:
            response = await llm_chat_engine.astream_chat(query.question)
        else:
            response = await llm_chat_engine.achat(query.question)
        self.chat_store.persist()
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

        response = await self.agent.achat(query.question)
        return LlmResponse(answer=response.response)

    async def batch_evaluate_retrieval_and_response(self, type):
        batch_eval = BatchEvaluator(self.config, self.retriever, self.query_engine)
        df, eval_res_avg = await batch_eval.batch_retrieval_response_aevaluation(
            type=type, workers=2, save_to_file=True
        )

        return df, eval_res_avg
