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
)

from pai_rag.app.api.models import (
    RagQuery,
    ContextDoc,
    RetrievalResponse,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import ImageNode
from loguru import logger
from enum import Enum

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
        self.chat_flow = ChatFlow()
        vector_index = resolve_vector_index(knowledgebase_manager.get_knowledgebase())
        _ = resolve_query_engine(self.config, vector_index=vector_index)

    def refresh(self, config: RagConfig):
        self.config = config

    async def achat(self, chat_request: ChatCompletionRequest):
        return await self.chat_flow.achat(chat_request, self.config)

    async def astream_chat(self, chat_request: ChatCompletionRequest):
        return await self.chat_flow.astream_chat(chat_request, self.config)

    async def aquery(
        self,
        query: RagQuery,
        chat_type: RagChatType = RagChatType.RAG,
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

        chat_knowledgebase = False
        search_web = False
        chat_agent = False
        chat_db = False
        chat_llm = False

        if query.with_intent:
            chat_type = RagChatType.Agent

        if chat_type == RagChatType.RAG:
            chat_knowledgebase = True
        elif chat_type == RagChatType.WEB:
            search_web = True
        elif chat_type == RagChatType.LLM:
            chat_llm = True
        elif chat_type == RagChatType.NL2SQL:
            chat_db = True
        elif chat_type == RagChatType.Agent:
            chat_agent = True

        chat_request = ChatCompletionRequest(
            messages=messages,
            stream=query.stream,
            model=query.model,
            index_name=query.index_name,
            chat_knowledgebase=chat_knowledgebase,
            search_web=search_web,
            chat_llm=chat_llm,
            chat_agent=chat_agent,
            chat_db=chat_db,
            return_reference=query.return_reference,
        )

        return await self.chat_flow.aquery(
            session_id=session_id,
            chat_request=chat_request,
            config=self.config,
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

    async def aload_db_info(self):
        db_info_loader = resolve_data_analysis_loader(self.config)
        await db_info_loader.aload_db_info()

        return "Load database info successfully."
