"""FastAPI dependency injection functions for services."""

from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
import os
from db.db_context import get_db_session
from service.model.llm_service import LlmService
from service.model.embedding_service import EmbeddingService
from service.model.reranker_service import RerankerService
from service.tool.websearch_service import WebsearchService
from service.tool.chatdb_service import ChatdbService
from service.tool.chatapp_service import ChatappService
from service.tool.faq_config_service import FAQConfigService
from service.tool.faq_item_service import FAQItemService
from service.tool.codesandbox_service import CodesandboxService
from service.tool.evaluation_service import EvaluationService
from service.tool.guardrail_service import GuardrailService
from service.tool.mcpserver_service import McpserverService
from service.tool.trace_service import TraceService
from service.knowledgebase.vectordb_service import VectordbService
from service.tool.role_service import RoleService
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from service.knowledgebase.datasource_service import DataSourceService
from service.knowledgebase.file_service import FileService
from service.file.file_resource_service import FileResourceService
from service.knowledgebase.chunk_service import ChunkService
from service.knowledgebase.metadata_service import MetadataService
from service.knowledgebase.file_metadata_relation_service import (
    FileMetadataRelationService,
)
from service.knowledgebase.file_task_service import FileTaskService
from service.knowledgebase.rag_service import RagService
from service.knowledgebase.vector_table_mapping_service import VectorTableMappingService
from service.thread.thread_service import ThreadService
from service.thread.message_service import MessageService
from service.agent.agent_service import AgentService
from service.model.bailian_model_service import BailianModelService

from fastapi import Header, HTTPException
from typing import Optional
from common.system_constants import ENABLE_TENANT_ID, DEFAULT_TENANT_ID

# 依赖项函数：从请求头中获取 Tenant ID
async def get_tenant_id(
    # 使用 FastAPI Header 依赖项来获取 HTTP 请求头 'X-Tenant-ID' 的值
    x_tenant_id: Optional[str] = Header(None, alias="X-TENANT-ID"),
) -> str:
    """
    从请求头中提取 X-TENANT-ID
    """
    if x_tenant_id is None:
        if ENABLE_TENANT_ID:
            raise HTTPException(
                status_code=400,
                detail="Tenant ID is missing in the X-TENANT-ID header."
            )
        else:
            # 未激活tenant id，则返回默认tenant id
            return DEFAULT_TENANT_ID

    return x_tenant_id


async def get_llm_service(
    session: AsyncSession = Depends(get_db_session),
) -> LlmService:
    """
    FastAPI dependency injection function for LlmService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        LlmService instance with the injected session

    Example:
        ```python
        @router.get("/llms")
        async def list_llms(
            llm_service: LlmService = Depends(get_llm_service),
        ):
            return await llm_service.list_llms()
        ```
    """
    if os.environ.get("USE_BAILIAN_MODEL_SERVICE", "false").lower() == "true":
        return BailianModelService()
    return LlmService(session)


async def get_embedding_service(
    session: AsyncSession = Depends(get_db_session),
) -> EmbeddingService:
    """
    FastAPI dependency injection function for EmbeddingService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        EmbeddingService instance with the injected session

    Example:
        ```python
        @router.get("/embeddings")
        async def list_embeddings(
            embedding_service: EmbeddingService = Depends(get_embedding_service),
        ):
            return await embedding_service.list_embeddings()
        ```
    """
    if os.environ.get("USE_BAILIAN_MODEL_SERVICE", "false").lower() == "true":
        return BailianModelService()
    return EmbeddingService(session)


async def get_reranker_service(
    session: AsyncSession = Depends(get_db_session),
) -> RerankerService:
    """
    FastAPI dependency injection function for RerankerService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        RerankerService instance with the injected session

    Example:
        ```python
        @router.get("/rerankers")
        async def list_rerankers(
            reranker_service: RerankerService = Depends(get_reranker_service),
        ):
            return await reranker_service.list_rerankers()
        ```
    """
    if os.environ.get("USE_BAILIAN_MODEL_SERVICE", "false").lower() == "true":
        return BailianModelService()
    return RerankerService(session)


async def get_websearch_service(
    session: AsyncSession = Depends(get_db_session),
) -> WebsearchService:
    """
    FastAPI dependency injection function for WebsearchService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        WebsearchService instance with the injected session

    Example:
        ```python
        @router.get("/websearch")
        async def get_websearch_config(
            websearch_service: WebsearchService = Depends(get_websearch_service),
        ):
            return await websearch_service.get_websearch_config_or_create()
        ```
    """
    return WebsearchService(session)


async def get_knowledgebase_service(
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgebaseService:
    """
    FastAPI dependency injection function for KnowledgebaseService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        KnowledgebaseService instance with the injected session

    Example:
        ```python
        @router.get("/knowledgebases")
        async def list_knowledgebases(
            kb_service: KnowledgebaseService = Depends(get_knowledgebase_service),
        ):
            return await kb_service.list_knowledgebases()
        ```
    """
    return KnowledgebaseService(session)


async def get_datasource_service(
    session: AsyncSession = Depends(get_db_session),
) -> DataSourceService:
    """
    FastAPI dependency injection function for DataSourceService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        DataSourceService instance with the injected session
    """
    return DataSourceService(session)


async def get_file_service(
    session: AsyncSession = Depends(get_db_session),
) -> FileService:
    """
    FastAPI dependency injection function for FileService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        FileService instance with the injected session

    Example:
        ```python
        @router.get("/{kb_id}/files")
        async def list_files(
            kb_id: str,
            file_service: FileService = Depends(get_file_service),
        ):
            return await file_service.list_files(kb_id=kb_id)
        ```
    """
    return FileService(session)


async def get_file_resource_service(
    session: AsyncSession = Depends(get_db_session),
) -> FileResourceService:
    """FastAPI DI for the new /v1/files FileResourceService (decoupled from KB)."""
    return FileResourceService(session)


async def get_chunk_service(
    session: AsyncSession = Depends(get_db_session),
) -> ChunkService:
    """
    FastAPI dependency injection function for ChunkService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        ChunkService instance with the injected session

    Example:
        ```python
        @router.get("/{kb_id}/files/{file_id}/chunks")
        async def list_chunks(
            kb_id: str,
            file_id: str,
            chunk_service: ChunkService = Depends(get_chunk_service),
        ):
            return await chunk_service.list_chunks(kb_id=kb_id, file_id=file_id)
        ```
    """
    return ChunkService(session)


async def get_metadata_service(
    session: AsyncSession = Depends(get_db_session),
) -> MetadataService:
    """
    FastAPI dependency injection function for MetadataService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        MetadataService instance with the injected session

    Example:
        ```python
        @router.get("/{kb_id}/metadata")
        async def list_metadata(
            kb_id: str,
            metadata_service: MetadataService = Depends(get_metadata_service),
        ):
            return await metadata_service.list_metadata(kb_id=kb_id)
        ```
    """
    return MetadataService(session)


async def get_file_task_service(
    session: AsyncSession = Depends(get_db_session),
) -> FileTaskService:
    """
    FastAPI dependency injection function for FileTaskService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        FileTaskService instance with the injected session

    Example:
        ```python
        @router.get("/{kb_id}/files/{file_id}/tasks")
        async def list_file_tasks(
            kb_id: str,
            file_id: str,
            file_task_service: FileTaskService = Depends(get_file_task_service),
        ):
            return await file_task_service.list_file_tasks(kb_id=kb_id, file_id=file_id)
        ```
    """
    return FileTaskService(session)


async def get_file_metadata_relation_service(
    session: AsyncSession = Depends(get_db_session),
) -> FileMetadataRelationService:
    """
    FastAPI dependency injection function for FileMetadataRelationService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        FileMetadataRelationService instance with the injected session

    Example:
        ```python
        @router.get("/{kb_id}/file-metadata-relations")
        async def list_file_metadata_relations(
            kb_id: str,
            file_metadata_relation_service: FileMetadataRelationService = Depends(get_file_metadata_relation_service),
        ):
            return await file_metadata_relation_service.get_file_metadata_relations(kb_id=kb_id)
        ```
    """
    return FileMetadataRelationService(session)


async def get_vector_table_mapping_service(
    session: AsyncSession = Depends(get_db_session),
) -> VectorTableMappingService:
    """
    FastAPI dependency injection function for VectorTableMappingService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        VectorTableMappingService instance with the injected session

    Example:
        ```python
        @router.get("/{kb_id}/vector-table-name")
        async def get_vector_table_name(
            kb_id: str,
            tenant_id: str,
            mapping_service: VectorTableMappingService = Depends(get_vector_table_mapping_service),
        ):
            return await mapping_service.get_vector_table_name(tenant_id, kb_id)
        ```
    """
    return VectorTableMappingService(session)


async def get_rag_service(
    session: AsyncSession = Depends(get_db_session),
) -> RagService:
    """
    FastAPI dependency injection function for RagService.

    RagService orchestrates business logic across knowledgebase entities.
    Use this for operations that involve multiple entities (e.g., deleting a knowledgebase
    with all its files, chunks, and metadata).

    Args:
        session: Database session (injected via Depends)

    Returns:
        RagService instance with injected session and service getters

    Example:
        ```python
        @router.delete("/{kb_id}")
        async def delete_knowledgebase(
            kb_id: str,
            rag_service: RagService = Depends(get_rag_service),
        ):
            await rag_service.delete_knowledgebase(kb_id)
        ```
    """
    # Create lazy getters for related services using nested async functions
    async def kb_service_getter(db_session: AsyncSession = session):
        return await get_knowledgebase_service(db_session)

    async def file_service_getter(db_session: AsyncSession = session):
        return await get_file_service(db_session)

    async def chunk_service_getter(db_session: AsyncSession = session):
        return await get_chunk_service(db_session)

    async def metadata_service_getter(db_session: AsyncSession = session):
        return await get_metadata_service(db_session)

    async def file_metadata_relation_service_getter(db_session: AsyncSession = session):
        return await get_file_metadata_relation_service(db_session)

    async def embedding_service_getter(db_session: AsyncSession = session):
        return await get_embedding_service(db_session)

    async def reranker_service_getter(db_session: AsyncSession = session):
        return await get_reranker_service(db_session)

    async def llm_service_getter(db_session: AsyncSession = session):
        return await get_llm_service(db_session)

    async def vector_db_service_getter(db_session: AsyncSession = session):
        return await get_vectordb_service(db_session)

    async def vector_table_mapping_service_getter(db_session: AsyncSession = session):
        return await get_vector_table_mapping_service(db_session)

    return RagService(
        session=session,
        kb_service_getter=kb_service_getter,
        file_service_getter=file_service_getter,
        chunk_service_getter=chunk_service_getter,
        metadata_service_getter=metadata_service_getter,
        file_metadata_relation_service_getter=file_metadata_relation_service_getter,
        embedding_service_getter=embedding_service_getter,
        reranker_service_getter=reranker_service_getter,
        llm_service_getter=llm_service_getter,
        vector_db_service_getter=vector_db_service_getter,
        vector_table_mapping_service_getter=vector_table_mapping_service_getter,
    )


async def get_chatdb_service(
    session: AsyncSession = Depends(get_db_session),
) -> ChatdbService:
    """
    FastAPI dependency injection function for ChatdbService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        ChatdbService instance with the injected session
    """
    return ChatdbService(session)


async def get_chatapp_service(
    session: AsyncSession = Depends(get_db_session),
) -> ChatappService:
    """
    FastAPI dependency injection function for ChatappService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        ChatappService instance with the injected session
    """
    return ChatappService(session)


async def get_faq_config_service(
    session: AsyncSession = Depends(get_db_session),
) -> FAQConfigService:
    """
    FastAPI dependency injection function for FAQConfigService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        FAQConfigService instance with the injected session
    """
    return FAQConfigService(session)


async def get_faq_item_service(
    session: AsyncSession = Depends(get_db_session),
) -> FAQItemService:
    """
    FastAPI dependency injection function for FAQItemService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        FAQItemService instance with the injected session
    """
    return FAQItemService(session)


async def get_codesandbox_service(
    session: AsyncSession = Depends(get_db_session),
) -> CodesandboxService:
    """
    FastAPI dependency injection function for CodesandboxService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        CodesandboxService instance with the injected session
    """
    return CodesandboxService(session)


async def get_evaluation_service(
    session: AsyncSession = Depends(get_db_session),
) -> EvaluationService:
    """
    FastAPI dependency injection function for EvaluationService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        EvaluationService instance with the injected session
    """
    return EvaluationService(session)


async def get_guardrail_service(
    session: AsyncSession = Depends(get_db_session),
) -> GuardrailService:
    """
    FastAPI dependency injection function for GuardrailService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        GuardrailService instance with the injected session
    """
    return GuardrailService(session)


async def get_mcpserver_service(
    session: AsyncSession = Depends(get_db_session),
) -> McpserverService:
    """
    FastAPI dependency injection function for McpserverService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        McpserverService instance with the injected session
    """
    return McpserverService(session)


async def get_trace_service(
    session: AsyncSession = Depends(get_db_session),
) -> TraceService:
    """
    FastAPI dependency injection function for TraceService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        TraceService instance with the injected session
    """
    return TraceService(session)


async def get_vectordb_service(
    session: AsyncSession = Depends(get_db_session),
) -> VectordbService:
    """
    FastAPI dependency injection function for VectordbService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        VectordbService instance with the injected session
    """
    return VectordbService(session)


async def get_role_service(
    session: AsyncSession = Depends(get_db_session),
) -> RoleService:
    """
    FastAPI dependency injection function for RoleService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        RoleService instance with the injected session

    Example:
        ```python
        @router.get("/roles")
        async def list_roles(
            role_service: RoleService = Depends(get_role_service),
        ):
            return await role_service.list_roles()
        ```
    """
    return RoleService(session)


async def get_thread_service(
    session: AsyncSession = Depends(get_db_session),
) -> ThreadService:
    """
    FastAPI dependency injection function for ThreadService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        ThreadService instance with the injected session

    Example:
        ```python
        @router.get("/threads")
        async def list_threads(
            thread_service: ThreadService = Depends(get_thread_service),
        ):
            return await thread_service.list_threads()
        ```
    """
    return ThreadService(session)


async def get_agent_service(
    session: AsyncSession = Depends(get_db_session),
) -> "AgentService":
    """
    FastAPI dependency injection function for AgentService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        AgentService instance with the injected session
    """
    # Create lazy getters for related services using nested async functions
    async def llm_service_getter():
        return await get_llm_service(session)

    async def chatapp_service_getter():
        return await get_chatapp_service(session)

    async def websearch_service_getter():
        return await get_websearch_service(session)

    async def mcpserver_service_getter():
        return await get_mcpserver_service(session)

    async def codesandbox_service_getter():
        return await get_codesandbox_service(session)

    async def chatdb_service_getter():
        return await get_chatdb_service(session)

    async def rag_service_getter():
        return await get_rag_service(session)

    async def file_resource_service_getter():
        return await get_file_resource_service(session)

    async def faq_config_service_getter():
        return await get_faq_config_service(session)

    return AgentService(
        session=session,
        llm_service_getter=llm_service_getter,
        chatapp_service_getter=chatapp_service_getter,
        websearch_service_getter=websearch_service_getter,
        mcpserver_service_getter=mcpserver_service_getter,
        codesandbox_service_getter=codesandbox_service_getter,
        chatdb_service_getter=chatdb_service_getter,
        rag_service_getter=rag_service_getter,
        file_resource_service_getter=file_resource_service_getter,
        faq_config_service_getter=faq_config_service_getter,
    )


async def get_message_service(
    session: AsyncSession = Depends(get_db_session),
) -> MessageService:
    """
    FastAPI dependency injection function for MessageService.

    Args:
        session: Database session (injected via Depends)

    Returns:
        MessageService instance with the injected session

    Example:
        ```python
        @router.get("/{thread_id}/messages")
        async def list_messages(
            thread_id: str,
            message_service: MessageService = Depends(get_message_service),
        ):
            return await message_service.list_messages(thread_id=thread_id)
        ```
    """
    return MessageService(session)
