import traceback
from pairag.chat.chat_context import set_context
from pairag.chat.chat_flow import ChatFlow
from pairag.core.rag_config import RagConfig
from pairag.integrations.query_transform.intent_models import IntentResult
from pairag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pairag.core.rag_module import (
    resolve_data_analysis_loader,
    resolve_vector_index,
)

from pairag.chat.models import (
    EmbeddingInput,
    RetrievalRequest,
    DocRecord,
    NewRetrievalResponse,
    ChatCompletionRequest,
)
from llama_index.core.schema import QueryBundle
from loguru import logger
from enum import Enum
from pairag.integrations.trace.base import init_instrument
from openai.types.create_embedding_response import CreateEmbeddingResponse
from pairag.integrations.trace import context as trace_context

DEFAULT_RAG_INDEX_FILE = "localdata/default_rag_indexes.json"


class RagChatType(str, Enum):
    LLM = "llm"
    RAG = "rag"
    WEB = "web"
    NL2SQL = "nl2sql"
    Agent = "agent"


class ChatApp:
    def __init__(self, config: RagConfig):
        self.config = config
        init_instrument(self.config.trace)
        self.trace_key_maps = self.config.trace.user_args or {}

        try:
            _ = resolve_vector_index(knowledgebase_manager.get_knowledgebase())
        except Exception as ex:
            logger.warning(f"Try to init vector index failed: {traceback.format_exc()}")
            pass

    def refresh(self, config: RagConfig):
        self.config = config
        init_instrument(self.config.trace)
        self.trace_key_maps = self.config.trace.user_args or {}

    async def achat(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        trace_args = chat_request.trace_args or {}
        set_context(trace_args)

        for trace_key, trace_value in trace_args.items():
            telementry_key = self.trace_key_maps.get(trace_key)
            if telementry_key:
                trace_context.set_context_var(telementry_key, trace_value)

        return await chat_flow.achat(chat_request)

    async def astream_chat(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        # set user attributes here
        trace_args = chat_request.trace_args or {}
        set_context(trace_args)

        for trace_key, trace_value in trace_args.items():
            telementry_key = self.trace_key_maps.get(trace_key)
            if telementry_key:
                trace_context.set_context_var(telementry_key, trace_value)

        return await chat_flow.astream_chat(chat_request)

    async def aknowledgebase_retrieval(
        self, retrieval_request: RetrievalRequest
    ) -> NewRetrievalResponse:
        logger.info(
            f"aknowledgebase_retrieval ==> query: {retrieval_request.query} to knowledgebase_id: {retrieval_request.knowledgebase_id}"
        )

        chat_flow = ChatFlow(self.config)
        node_results = await chat_flow.aretrieve(
            query_str=retrieval_request.query,
            knowledgebase_name=retrieval_request.knowledgebase_id,
            extra_retrieval_settings=retrieval_request.retrieval_settings)

        records = [
            DocRecord(
                content=score_node.node.get_content(),
                score=score_node.score,
                title=score_node.node.metadata.get("file_name", "null"),
                metadata=score_node.node.metadata,
            )
            for score_node in node_results
        ]

        return NewRetrievalResponse(records=records)

    async def aload_db_info(self):
        db_info_loader = resolve_data_analysis_loader(self.config)
        await db_info_loader.aload_db_info()

        return "Load database info successfully."


    ## 原子能力调用
    async def achat_llm_atomic(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.achat_llm_atomic(chat_request)

    async def achat_web_atomic(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.achat_web_atomic(chat_request)

    async def achat_knowledgebase_atomic(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.achat_knowledgebase_atomic(chat_request)

    async def achat_news_agent_atomic(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.achat_news_agent_atomic(chat_request)

    async def astream_llm_atomic(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.astream_llm_atomic(chat_request)

    async def astream_web_atomic(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.astream_web_atomic(chat_request)

    async def astream_knowledgebase_atomic(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.astream_knowledgebase_atomic(chat_request)

    async def astream_news_agent_atomic(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.astream_news_agent_atomic(chat_request)

    async def arecognize_intent(self, chat_request: ChatCompletionRequest) -> IntentResult:
        chat_flow = ChatFlow(self.config)
        return await chat_flow.arecognize_intent(chat_request)

    async def aembed(self, embedding_input: EmbeddingInput) -> CreateEmbeddingResponse:
        chat_flow = ChatFlow(self.config)
        return await chat_flow.aembed(embedding_input)
