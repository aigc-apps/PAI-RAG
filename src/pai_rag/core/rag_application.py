from pai_rag.app.api.models import (
    ChatCompletionRequest,
)
from pai_rag.core.chat_flow import ChatFlow
from pai_rag.core.rag_config import RagConfig
from pai_rag.core.utils.chat_utils import SseVersion, chat_id_generator
from pai_rag.knowledgebase.rag_knowledgebase import knowledgebase_manager
from pai_rag.core.rag_module import (
    resolve_chat_store,
    resolve_data_analysis_loader,
    resolve_query_engine,
    resolve_vector_index,
    resolve_query_engine_from_retrieval_request,
)

from pai_rag.app.api.models import (
    RagQuery,
    ContextDoc,
    RetrievalResponse,
    RetrievalRequest,
    DocRecord,
    NewRetrievalResponse,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import ImageNode
from loguru import logger
from enum import Enum
from pai_rag.integrations.trace.base import init_trace
from pai_rag.utils.messages_utils import parse_chat_messages_v2

DEFAULT_RAG_INDEX_FILE = "localdata/default_rag_indexes.json"


class RagChatType(str, Enum):
    LLM = "llm"
    RAG = "rag"
    WEB = "web"
    NL2SQL = "nl2sql"
    Agent = "agent"


class PaiApp:
    def __init__(self, config: RagConfig):
        self.config = config
        if self.config.trace.is_enabled():
            init_trace(self.config.trace)

        vector_index = resolve_vector_index(knowledgebase_manager.get_knowledgebase())
        _ = resolve_query_engine(self.config, vector_index=vector_index)

    def refresh(self, config: RagConfig):
        self.config = config
        if self.config.trace.is_enabled():
            init_trace(self.config.trace)

    async def achat(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.achat(chat_request)

    async def astream_chat(self, chat_request: ChatCompletionRequest):
        chat_flow = ChatFlow(self.config)
        return await chat_flow.astream_chat(chat_request)

    async def aquery(
        self,
        query: RagQuery,
        chat_type: RagChatType,
        sse_version: SseVersion = SseVersion.V1,
    ):
        session_id = query.session_id or chat_id_generator()
        logger.debug(f"Get session ID: {session_id}.")

        chat_store = resolve_chat_store(self.config)
        messages = query.messages
        if messages is None or len(messages) == 0:
            messages = parse_chat_messages_v2(
                question=query.question,
                session_id=session_id,
                chat_history=query.chat_history,
                chat_store=chat_store,
            )
        if chat_type is None:
            chat_knowledgebase = query.chat_knowledgebase
            search_web = query.search_web
            chat_agent = query.chat_agent
            chat_db = query.chat_db
            chat_llm = query.chat_llm
        else:
            if query.with_intent:
                chat_type = RagChatType.Agent

            chat_knowledgebase = chat_type == RagChatType.RAG
            search_web = chat_type == RagChatType.WEB
            chat_llm = chat_type == RagChatType.LLM
            chat_db = chat_type == RagChatType.NL2SQL
            chat_agent = chat_type == RagChatType.Agent

        chat_request = ChatCompletionRequest(
            messages=messages,
            stream=query.stream,
            model=query.model or "default",
            index_name=query.index_name,
            chat_knowledgebase=chat_knowledgebase,
            search_web=search_web,
            chat_llm=chat_llm,
            chat_agent=chat_agent,
            chat_db=chat_db,
            return_reference=query.return_reference,
            temperature=query.temperature,
        )

        chat_flow = ChatFlow(self.config)
        return await chat_flow.aquery(
            session_id=session_id,
            chat_request=chat_request,
            chat_store=chat_store,
            sse_version=sse_version,
        )

    async def aretrieve(
        self,
        question: str,
        knowledgebase_name: str = None,
    ) -> RetrievalResponse:
        query_bundle = QueryBundle(question)
        knowledgebase = knowledgebase_manager.get_knowledgebase(knowledgebase_name)
        vector_index = resolve_vector_index(knowledgebase=knowledgebase)
        query_engine = resolve_query_engine(self.config, vector_index=vector_index)
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

    async def aknowledgebase_retrieval(
        self, retrieval_request: RetrievalRequest
    ) -> NewRetrievalResponse:
        query_bundle = QueryBundle(retrieval_request.query)
        knowledgebase = knowledgebase_manager.get_knowledgebase(
            retrieval_request.knowledgebase_id
        )
        vector_index = resolve_vector_index(knowledgebase=knowledgebase)
        _retrieval_settings = {
            **knowledgebase.retrieval_settings,
            **retrieval_request.retrieval_settings,
        }
        logger.info(
            f"aknowledgebase_retrieval ==> query: {retrieval_request.query} to knowledgebase_id: {retrieval_request.knowledgebase_id} with retrieval_settings: {_retrieval_settings}"
        )
        query_engine = resolve_query_engine_from_retrieval_request(
            self.config,
            vector_index=vector_index,
            retrieval_settings=_retrieval_settings,
        )
        node_results = await query_engine.aretrieve(query_bundle)

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
