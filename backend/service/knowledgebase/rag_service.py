"""RAG Service layer for orchestrating business logic across knowledgebase entities."""

from typing import Callable, Awaitable, Optional, List, Tuple
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm.attributes import flag_modified
import asyncio
import copy
from common.chat.response_model import PagedResult
from common.chat.models import RetrievalSetting, MetadataFilteringCondition

from db.models.knowledgebase.knowledgebase import (
    KbEntity,
    KnowledgebaseCreate,
    RetrievalConfig,
)
from db.models.knowledgebase.file import KbFileEntity, MetadataEntryData
from db.models.knowledgebase.metadata import (
    KbMetadataEntity,
    KbMetadataEntityCreate,
)
from db.models.knowledgebase.embedding import EmbeddingModelEntity
from db.models.knowledgebase.chunk import KbChunkEntity, create_text_node_from_chunk
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from service.knowledgebase.utils.metadata_utils import validate_metadata_value
from service.knowledgebase.datasource_service import DataSourceService
from service.factory.model_factory import create_reranker_model, create_embedding_model
from rag.metadata_filter import EmptyFilesException, query_file_ids_with_metadata_filter
from service.factory.vectordb_factory import create_vector_store
from rag.vector_store.vector_connection import is_docid_filter_supported
from rag.rerank.fusion_reranker import arerank_fusion
from llama_index.core.schema import BaseNode
from pairag.file.store.file_store_helper import file_store
from common.knowledgebase.constants import DEFAULT_METADATA_KEYS
from rag.parse_utils import get_node_texts_for_embedding
from tools.utils.vectordb_retrieval import retrieval_type_to_search_mode
import re
from common.tool.search_result import SearchResult
from tools.utils.vectordb_retrieval import aquery_vector_store
from utils.lru_cache import LruCache
from db.db_context import create_db_session
from common.knowledgebase.constants import DEFAULT_VECTOR_WEIGHT, DEFAULT_SIMILARITY_TOP_K, DEFAULT_RERANK_SIMILARITY_TOP_K
from extensions.trace.rag_wrapper import query_knowledgebase_wrapper, embedding_wrapper
from openinference.instrumentation import suppress_tracing
from loguru import logger

MARKDOWN_IMAGE_PATTERN = r'!\[.*?\]\((.*?)\)\s*\n*\s*图片的描述:\s*(.*?)(?=\n\n|$)'


class RagService:
    """
    Service layer for orchestrating business logic across knowledgebase entities.

    This service coordinates operations that involve multiple entities (knowledgebase, file, chunk, metadata)
    by delegating to specialized services for individual entity CRUD operations.
    """

    def __init__(
        self,
        session: AsyncSession,
        kb_service_getter: Callable[[AsyncSession], Awaitable],
        file_service_getter: Callable[[AsyncSession], Awaitable],
        chunk_service_getter: Callable[[AsyncSession], Awaitable],
        metadata_service_getter: Callable[[AsyncSession], Awaitable],
        file_metadata_relation_service_getter: Callable[[AsyncSession], Awaitable],
        embedding_service_getter: Callable[[AsyncSession], Awaitable],
        reranker_service_getter: Callable[[AsyncSession], Awaitable],
        llm_service_getter: Callable[[AsyncSession], Awaitable],
        vector_db_service_getter: Callable[[AsyncSession], Awaitable],
        vector_table_mapping_service_getter: Callable[[AsyncSession], Awaitable],
    ):
        """
        Initialize RagService with a database session and service getters.

        Args:
            session: Database session (injected dependency)
            kb_service_getter: Async callable that returns KnowledgebaseService instance (required)
            file_service_getter: Async callable that returns FileService instance (required)
            chunk_service_getter: Async callable that returns ChunkService instance (required)
            metadata_service_getter: Async callable that returns MetadataService instance (required)
            file_metadata_relation_service_getter: Async callable that returns FileMetadataRelationService instance (required)
            embedding_service_getter: Async callable that returns EmbeddingService instance (required)
            reranker_service_getter: Async callable that returns RerankerService instance (required)
            llm_service_getter: Async callable that returns LlmService instance (required)
            vector_db_service_getter: Async callable that returns VectordbService instance (required)
            vector_table_mapping_service_getter: Async callable that returns VectorTableMappingService instance (required)
        Raises:
            ValueError: If any getter is None
        """
        if kb_service_getter is None:
            raise ValueError("kb_service_getter is required")
        if file_service_getter is None:
            raise ValueError("file_service_getter is required")
        if chunk_service_getter is None:
            raise ValueError("chunk_service_getter is required")
        if metadata_service_getter is None:
            raise ValueError("metadata_service_getter is required")
        if file_metadata_relation_service_getter is None:
            raise ValueError("file_metadata_relation_service_getter is required")
        if embedding_service_getter is None:
            raise ValueError("embedding_service_getter is required")
        if reranker_service_getter is None:
            raise ValueError("reranker_service_getter is required")
        if llm_service_getter is None:
            raise ValueError("llm_service_getter is required")
        if vector_db_service_getter is None:
            raise ValueError("vector_db_service_getter is required")
        if vector_table_mapping_service_getter is None:
            raise ValueError("vector_table_mapping_service_getter is required")

        self.session = session
        self._get_kb_service = kb_service_getter
        self._get_file_service = file_service_getter
        self._get_chunk_service = chunk_service_getter
        self._get_metadata_service = metadata_service_getter
        self._get_file_metadata_relation_service = file_metadata_relation_service_getter
        self._get_embedding_service = embedding_service_getter
        self._get_reranker_service = reranker_service_getter
        self._get_llm_service = llm_service_getter
        self._get_vector_db_service = vector_db_service_getter
        self._get_vector_table_mapping_service = vector_table_mapping_service_getter
        self._embed_dimension_cache = LruCache(max_size=100)

    # ------------------------------------------------------------------------------------------------
    # Knowledgebase operations: list, add, get, update, delete
    # ------------------------------------------------------------------------------------------------

    async def list_knowledgebases(
        self,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
        query: Optional[str] = None,
    ) -> PagedResult[List[KbEntity]]:
        kb_service = await self._get_kb_service()
        return await kb_service.list_knowledgebases(
            tenant_id=tenant_id,
            page=page,
            size=size,
            query=query,
        )


    async def _validate_knowledgebase_models(
        self, kb_data: KnowledgebaseCreate, tenant_id: str
    ) -> None:
        """
        Validate that all referenced models (embedding, reranker, image_caption) exist and are valid.

        Args:
            knowledgebase: Knowledgebase data to validate

        Raises:
            ValueError: If any model is invalid or doesn't exist
        """
        # Validate embedding_model
        logger.info(f"Validating knowledgebase models: {kb_data}")
        if not kb_data.embedding_model:
            raise ValueError("Cannot create knowledgebase without embedding model.")

        embedding_service = await self._get_embedding_service()
        embedding_model = await embedding_service.get_embedding_model_by_provider_model_id(
            provider_name=kb_data.embedding_provider_name,
            model_id=kb_data.embedding_model,
            tenant_id=tenant_id,
        )
        if not embedding_model:
            raise ValueError(
                f"Embedding model '{kb_data.embedding_model}' does not exist."
            )

        # Validate rerank_model if rerank is enabled
        if kb_data.retrieval_config:
            retrieval_config = kb_data.retrieval_config
            if retrieval_config.enable_rerank:
                if not retrieval_config.rerank_model:
                    raise ValueError("Cannot enable reranking without rerank model.")
                reranker_service = await self._get_reranker_service()
                reranker_model = await reranker_service.get_reranker_model_by_provider_model_id(
                    provider_name=retrieval_config.rerank_provider_name,
                    model_id=retrieval_config.rerank_model,
                    tenant_id=tenant_id,
                )
                if not reranker_model:
                    raise ValueError(
                        f"Rerank model '{retrieval_config.rerank_model}' does not exist."
                    )

        # Validate image_caption_model if specified
        if kb_data.chunk_config and kb_data.chunk_config.image_caption_model:
            llm_service = await self._get_llm_service()
            llm_model = await llm_service.get_llm_model_by_provider_model_id(
                provider_name=kb_data.chunk_config.image_caption_provider_name,
                model_id=kb_data.chunk_config.image_caption_model,
                tenant_id=tenant_id,
            )
            if not llm_model:
                raise ValueError(
                    f"Image caption model '{kb_data.chunk_config.image_caption_model}' does not exist."
                )
            if not llm_model.vision_support:
                raise ValueError(
                    f"Image caption model '{kb_data.chunk_config.image_caption_model}' does not support vision functionality."
                )

    async def create_knowledgebase(
        self, kb_data: KnowledgebaseCreate, tenant_id: str
    ) -> KbEntity:
        """
        Add a knowledgebase.
        This orchestrates the addition across knowledgebase, file, chunk, metadata, and file metadata relation services.

        Args:
            knowledgebase: Knowledgebase data

        Returns:
            Created KbEntity

        Raises:
            ValueError: If any referenced model is invalid or doesn't exist
        """
        # Validate all referenced models
        await self._validate_knowledgebase_models(kb_data, tenant_id=tenant_id)

        # Create knowledgebase
        kb_service = await self._get_kb_service()
        return await kb_service.create_knowledgebase(kb_data=kb_data, tenant_id=tenant_id)


    async def update_knowledgebase(
        self, kb_id: str, update_data: KnowledgebaseCreate, tenant_id: str
    ) -> KbEntity:
        """
        Update a knowledgebase.
        This orchestrates the update across knowledgebase, file, chunk, metadata, and file metadata relation services.

        Args:
            kb_id: Knowledgebase ID
            knowledgebase: Knowledgebase data

        Returns:
            Updated KbEntity

        Raises:
            ValueError: If any referenced model is invalid or doesn't exist
        """
        # Get existing knowledgebase to merge with update data
        kb_service = await self._get_kb_service()
        # Validate all referenced models (including existing ones if not being updated)
        await self._validate_knowledgebase_models(update_data, tenant_id=tenant_id)

        # Update knowledgebase
        return await kb_service.update_knowledgebase(kb_id=kb_id, update_data=update_data, tenant_id=tenant_id)

    async def get_knowledgebase_by_name(self, name: str, tenant_id: str) -> Optional[KbEntity]:
        """
        Get a knowledgebase by name.
        This orchestrates the retrieval across knowledgebase services.

        Args:
            name: Knowledgebase name
        """
        kb_service = await self._get_kb_service()
        return await kb_service.get_knowledgebase_by_name(name=name, tenant_id=tenant_id)

    async def get_knowledgebase(self, kb_id: str, tenant_id: str) -> Optional[KbEntity]:
        """
        Get a knowledgebase.
        This orchestrates the retrieval across knowledgebase, file, chunk, metadata, and file metadata relation services.

        Args:
            kb_id: Knowledgebase ID

        Returns:
            KbEntity if found, None otherwise
        """
        kb_service = await self._get_kb_service()
        return await kb_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)

    async def delete_knowledgebase(self, kb_id: str, tenant_id: str) -> None:
        """
        Delete a knowledgebase and all related entities (files, chunks, metadata).
        This orchestrates the deletion across multiple services.

        Args:
            kb_id: Knowledgebase ID

        Raises:
            ValueError: If knowledgebase not found
        """
        kb_service = await self._get_kb_service()
        knowledgebase = await kb_service.get_knowledgebase(kb_id, tenant_id=tenant_id)
        if not knowledgebase:
            raise ValueError(f"Knowledgebase '{kb_id}' does not exist.")

        # Delete related files directly (no need to query first)
        file_service = await self._get_file_service()
        await file_service.delete_files_from_kb(kb_id=kb_id, tenant_id=tenant_id)

        # Delete related chunks directly (no need to query first)
        chunk_service = await self._get_chunk_service()
        chunk_ids = await chunk_service.delete_chunks_from_kb(kb_id=kb_id, tenant_id=tenant_id)
        await self.adelete(kb_id=kb_id, node_ids=chunk_ids, tenant_id=tenant_id)

        # Delete related metadata in batch
        metadata_service = await self._get_metadata_service()
        await metadata_service.delete_metadata_by_kb_id(kb_id=kb_id, tenant_id=tenant_id)

        # Delete related file metadata relations
        file_metadata_relation_service = await self._get_file_metadata_relation_service()
        await file_metadata_relation_service.delete_file_metadata_relations_by_kb_id(kb_id=kb_id, tenant_id=tenant_id)

        # Delete knowledgebase itself
        await kb_service.delete_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)

        logger.info(f"Deleted knowledgebase {kb_id} and all related entities")


    # ------------------------------------------------------------------------------------------------
    # File operations: get, add, update, delete
    # ------------------------------------------------------------------------------------------------

    async def list_files(
        self,
        kb_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
        query: Optional[str] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
    ) -> PagedResult[List[KbFileEntity]]:
        """
        List files in a knowledgebase.

        Args:
            kb_id: Knowledgebase ID
            page: Page number (1-indexed)
            size: Page size
            query: Optional search query (searches in file_name + title)
            status: Optional filter for file status
            source: Optional origin filter ("manual" or a datasource_key)

        Returns:
            PagedResult containing list of KbFileEntity and pagination metadata
        """
        file_service = await self._get_file_service()
        return await file_service.list_files(kb_id=kb_id, tenant_id=tenant_id, page=page, size=size, query=query, status=status, source=source)

    async def keyword_search(
        self, kb_id: str, tenant_id: str, pattern: str,
        doc_id: Optional[str] = None, path_prefix: Optional[str] = None,
        datasource: Optional[str] = None, context: int = 2, limit: int = 20,
    ) -> dict:
        """Literal (non-regex) keyword grep over document bodies.

        The unfiltered case searches the whole knowledge base (manual uploads
        included). ``doc_id`` / ``path_prefix`` / ``datasource`` still narrow to
        specific data-source documents when supplied.

        Bounded for safety: a SQL/manifest prefilter picks candidate files, then
        each is grepped line-by-line from the file store for accurate line numbers
        + surrounding context. Returns ``{results, scanned_files, scan_capped,
        limit_reached}`` where each result is ``{doc_id, file_id, line, match,
        context, source_url, title}``.
        """
        from sqlalchemy import and_
        from db.models.knowledgebase.datasource import DataSourceDocumentEntity, DataSourceEntity

        pattern = (pattern or "").strip()
        if not pattern:
            return {"results": [], "scanned_files": 0, "scan_capped": False, "limit_reached": False}

        MAX_SCAN_FILES = 200
        file_ids: List[str] = []
        scan_capped = False

        if doc_id:
            r = await self.session.exec(
                select(DataSourceDocumentEntity.file_id).where(
                    DataSourceDocumentEntity.kb_id == kb_id,
                    DataSourceDocumentEntity.doc_id == doc_id,
                    DataSourceDocumentEntity.tenant_id == tenant_id,
                )
            )
            fid = r.first()
            file_ids = [fid] if fid else []
        elif path_prefix or datasource:
            conds = [
                DataSourceDocumentEntity.kb_id == kb_id,
                DataSourceDocumentEntity.tenant_id == tenant_id,
                DataSourceDocumentEntity.file_id.is_not(None),
            ]
            if path_prefix:
                conds.append(DataSourceDocumentEntity.path.like(f"{path_prefix}%"))
            if datasource:
                ds_ids = (await self.session.exec(
                    select(DataSourceEntity.id).where(
                        DataSourceEntity.kb_id == kb_id,
                        DataSourceEntity.datasource_key == datasource,
                        DataSourceEntity.tenant_id == tenant_id,
                    )
                )).all()
                conds.append(DataSourceDocumentEntity.datasource_id.in_(list(ds_ids) or ["__none__"]))
            rows = (await self.session.exec(
                select(DataSourceDocumentEntity.file_id).where(and_(*conds)).limit(MAX_SCAN_FILES)
            )).all()
            file_ids = [x for x in rows if x]
        else:
            # literal prefilter on chunk text across the whole KB. autoescape so
            # literal % / _ in the pattern are not treated as SQL LIKE wildcards
            # (which would broaden the prefilter and could crowd out real matches
            # under the MAX_SCAN_FILES cap).
            rows = (await self.session.exec(
                select(KbChunkEntity.file_id).where(
                    KbChunkEntity.kb_id == kb_id,
                    KbChunkEntity.tenant_id == tenant_id,
                    KbChunkEntity.text.contains(pattern, autoescape=True),
                ).distinct().limit(MAX_SCAN_FILES)
            )).all()
            file_ids = [x for x in rows if x]
            scan_capped = len(file_ids) >= MAX_SCAN_FILES

        if not file_ids:
            return {"results": [], "scanned_files": 0, "scan_capped": scan_capped, "limit_reached": False}

        entities = {}
        eres = await self.session.exec(
            select(KbFileEntity).where(KbFileEntity.id.in_(file_ids), KbFileEntity.tenant_id == tenant_id)
        )
        for e in eres.all():
            entities[e.id] = e

        results: List[dict] = []
        done = False
        for fid in file_ids:
            if done:
                break
            entity = entities.get(fid)
            if not entity or not entity.file_content:
                continue
            text = entity.file_content
            lines = text.splitlines()
            md = entity.file_metadata or {}
            for idx, line in enumerate(lines):
                if pattern in line:
                    lo = max(0, idx - context)
                    hi = min(len(lines), idx + context + 1)
                    results.append({
                        "doc_id": md.get("source_doc_id") or fid,
                        "file_id": fid,
                        "line": idx + 1,
                        "match": line.strip()[:300],
                        "context": "\n".join(lines[lo:hi]),
                        "source_url": entity.file_source,
                        "title": md.get("title") or entity.file_name,
                    })
                    if len(results) >= limit:
                        done = True
                        break

        return {
            "results": results,
            "scanned_files": len(file_ids),
            "scan_capped": scan_capped,
            "limit_reached": len(results) >= limit,
        }

    async def get_file_content(
        self, kb_id: str, file_id: str, tenant_id: str,
        max_chars: Optional[int] = None, offset: int = 0,
    ) -> Optional[dict]:
        """Return a file's text (windowed) + metadata (powers fetch/查看文件).

        Prefers DB ``file_content`` (fast, no OSS round-trip). Falls back to OSS
        for legacy files ingested before file_content was populated, then to
        chunk reassembly.
        """
        file_service = await self._get_file_service()
        entity = await file_service.get_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
        if not entity:
            return None

        content = entity.file_content or None
        degraded = None

        # Fallback: reassemble from chunks (for legacy files with no file_content)
        if not content:
            chunk_service = await self._get_chunk_service()
            chunks = await chunk_service.get_chunks_by_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
            if chunks:
                content = "\n\n".join(c.text for c in sorted(chunks, key=lambda c: c.index) if c.text)
                degraded = "content_from_chunks"

        full = content or ""
        total = len(full)
        offset = max(0, offset or 0)
        if max_chars and max_chars > 0:
            windowed = full[offset:offset + max_chars]
        else:
            windowed = full[offset:] if offset else full
        returned = len(windowed)
        truncated = (offset + returned) < total

        md = entity.file_metadata or {}
        return {
            "file_id": file_id,
            "file_name": entity.file_name,
            "title": md.get("title") or entity.file_name,
            "source_url": entity.file_source or md.get("source_url"),
            "doc_id": md.get("source_doc_id"),
            "content": windowed,
            "content_length": total,
            "offset": offset,
            "returned_chars": returned,
            "truncated": truncated,
            "next_offset": (offset + returned) if truncated else None,
            "degraded": degraded,
            "metadata": md,
        }


    async def get_files_by_names(self, kb_id: str, file_names: List[str], tenant_id: str) -> List[KbFileEntity]:
        """
        Get a file by path from a knowledgebase.
        This orchestrates the retrieval across file services.

        Args:
            kb_id: Knowledgebase ID
            file_path: File path
        """
        file_service = await self._get_file_service()
        return await file_service.get_files_by_names(kb_id=kb_id, file_names=file_names, tenant_id=tenant_id)


    async def get_file_by_name(self, kb_id: str, file_name: str, tenant_id: str) -> Optional[KbFileEntity]:
        """
        Get a file by name from a knowledgebase.
        This orchestrates the retrieval across file services.

        Args:
            kb_id: Knowledgebase ID
            file_name: File name
        """
        file_service = await self._get_file_service()
        return await file_service.get_file_by_name(kb_id=kb_id, file_name=file_name, tenant_id=tenant_id)


    async def get_file(self, kb_id: str, file_id: str, tenant_id: str) -> KbFileEntity:
        """
        Get a file from a knowledgebase.
        This orchestrates the retrieval across file and chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
        """
        file_service = await self._get_file_service()
        return await file_service.get_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)


    async def add_file(self, kb_id: str, file: KbFileEntity, tenant_id: str) -> KbFileEntity:
        """
        Add a file to a knowledgebase.
        This orchestrates the addition across file and chunk services.

        Args:
            kb_id: Knowledgebase ID
            file: File data
        """
        file_service = await self._get_file_service()
        return await file_service.add_file(kb_id=kb_id, file=file, tenant_id=tenant_id)

    ## Batch
    async def batch_delete_files(self, kb_id: str, file_ids: List[str], tenant_id: str) -> None:
        """
        Batch delete files and related chunks.
        This orchestrates the deletion across file and chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_ids: List of file IDs
        """
        for file_id in file_ids:
            await self.delete_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)


    async def get_file_id_source_map(
        self,
        file_ids: List[str],
        tenant_id: str,
    ):
        if not file_ids:
            return {}

        file_source_results = (await self.session.exec(
            select( KbFileEntity.id, KbFileEntity.file_source ).where(
                KbFileEntity.id.in_(file_ids),
                KbFileEntity.tenant_id == tenant_id,
            )
        )).all()
        logger.info(f"Get file_source_results: {file_source_results}")
        return {
            file_id: file_source
            for file_id, file_source in file_source_results
        }

    # Chunk operations: get, add, update, delete, list

    async def list_chunks(self, kb_id: str, file_id: str, tenant_id: str, page: int = 1, size: int = 10) -> PagedResult[List[KbChunkEntity]]:
        """
        List chunks in a file.
        This orchestrates the retrieval across chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
        """
        chunk_service = await self._get_chunk_service()
        return await chunk_service.list_chunks(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id, page=page, size=size)


    async def get_chunk(self, kb_id: str, file_id: str, chunk_id: str, tenant_id: str) -> KbChunkEntity:
        """
        Get a chunk from a file.
        This orchestrates the retrieval across chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            chunk_id: Chunk ID
        """
        chunk_service = await self._get_chunk_service()
        return await chunk_service.get_chunk(kb_id=kb_id, file_id=file_id, chunk_id=chunk_id, tenant_id=tenant_id)


    async def add_chunk(self, kb_id: str, file_id: str, text: str, tenant_id: str, chunk_metadata: dict = None) -> KbChunkEntity:
        """
        Add a chunk to a file.
        This orchestrates the addition across chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            text: Chunk text
            chunk_metadata: Chunk metadata
        """
        file_entity = await self.get_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
        if not file_entity:
            raise ValueError(f"File '{file_id}' does not exist.")

        chunk_metadata = chunk_metadata or file_entity.file_metadata

        chunk_service = await self._get_chunk_service()
        chunk = await chunk_service.create_chunk(kb_id=kb_id, file_id=file_id, text=text, tenant_id=tenant_id, chunk_metadata=chunk_metadata)

        kb_node = create_text_node_from_chunk(chunk)
        await self.ainsert(kb_id=kb_id, nodes=[kb_node], tenant_id=tenant_id)
        return chunk


    async def update_chunk(self, kb_id: str, file_id: str, chunk_id: str, chunk: KbChunkEntity, tenant_id: str) -> KbChunkEntity:
        """
        Update a chunk in a file.
        This orchestrates the update across chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            chunk_id: Chunk ID
            chunk: Chunk data
        """
        chunk_service = await self._get_chunk_service()
        kb_chunk = await chunk_service.update_chunk(kb_id=kb_id, file_id=file_id, chunk_id=chunk_id, tenant_id=tenant_id, new_chunk=chunk)
        kb_node = create_text_node_from_chunk(kb_chunk)

        await self.adelete(kb_id=kb_id, node_ids=[kb_chunk.id], tenant_id=tenant_id)
        if kb_chunk.active:
            await self.ainsert(kb_id=kb_id, nodes=[kb_node], tenant_id=tenant_id)
        return kb_chunk


    async def delete_chunk(self, kb_id: str, file_id: str, chunk_id: str, tenant_id: str) -> None:
        """
        Delete a chunk from a file.
        This orchestrates the deletion across chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            chunk_id: Chunk ID
        """
        await self.adelete(kb_id=kb_id, node_ids=[chunk_id], tenant_id=tenant_id)

        chunk_service = await self._get_chunk_service()
        await chunk_service.delete_chunk(kb_id=kb_id, file_id=file_id, chunk_id=chunk_id, tenant_id=tenant_id)

    # ------------------------------------------------------------------------------------------------
    # Metadata operations: list, get, add, update, delete
    # ------------------------------------------------------------------------------------------------
    async def list_metadata(self, kb_id: str, tenant_id: str, page: int = 1, size: int = 10) -> PagedResult[List[KbMetadataEntity]]:
        """
        List metadata in a knowledgebase.
        This orchestrates the retrieval across metadata services.

        Args:
            kb_id: Knowledgebase ID
        """
        metadata_service = await self._get_metadata_service()
        return await metadata_service.list_metadata(kb_id=kb_id, tenant_id=tenant_id, page=page, size=size)

    async def get_metadata(self, kb_id: str, metadata_id: str, tenant_id: str) -> Optional[KbMetadataEntity]:
        """
        Get a metadata from a knowledgebase.
        This orchestrates the retrieval across metadata services.

        Args:
            kb_id: Knowledgebase ID
            metadata_id: Metadata ID
        """
        metadata_service = await self._get_metadata_service()
        return await metadata_service.get_metadata(kb_id=kb_id, metadata_id=metadata_id, tenant_id=tenant_id)

    async def update_metadata(
        self,
        kb_id: str,
        metadata_id: str,
        update_data: KbMetadataEntityCreate,
        tenant_id: str,
    ) -> KbMetadataEntity:
        """
        Update metadata configuration and update all related file metadata.
        This orchestrates the update across metadata and file services.

        Args:
            kb_id: Knowledgebase ID
            metadata_id: Metadata ID
            update_data: Updated Metadata data

        Returns:
            Updated KbMetadataEntity

        Raises:
            ValueError: If metadata not found
        """
        metadata_service = await self._get_metadata_service()

        # Get current metadata to compare changes
        metadata_entity = await metadata_service.get_metadata(kb_id=kb_id, metadata_id=metadata_id, tenant_id=tenant_id)
        if not metadata_entity:
            raise ValueError(f"Metadata '{metadata_id}' does not exist.")

        old_name = metadata_entity.name
        old_value_type = metadata_entity.value_type
        new_name = update_data.name
        new_value_type = update_data.value_type

        # Update metadata entity itself
        updated_metadata = await metadata_service.update_metadata(
            kb_id=kb_id, metadata_id=metadata_id, update_data=update_data, tenant_id=tenant_id
        )

        # Find all files using this metadata
        file_metadata_relation_service = await self._get_file_metadata_relation_service()
        file_metadata_list = await file_metadata_relation_service.get_file_metadata_relations_by_metadata_id(
            kb_id=kb_id, metadata_id=metadata_id, tenant_id=tenant_id
        )
        file_ids = [relation.file_id for relation in file_metadata_list]
        if file_ids:
            file_service = await self._get_file_service()
            file_list = await file_service.get_files_by_ids(file_ids=file_ids, tenant_id=tenant_id)
            # If name changed, update all related files' file_metadata key names
            if old_name != new_name:
                logger.info(
                    f"Updating file's file_metadata key name: {old_name} -> {new_name}."
                )
                for file_entity in file_list:
                    if old_name in file_entity.file_metadata:
                        value = file_entity.file_metadata[old_name]
                        # If value_type also changed, validate and convert value
                        if old_value_type != new_value_type:
                            is_valid, converted_value = validate_metadata_value(
                                value, new_value_type
                            )
                            if not is_valid:
                                logger.warning(
                                    f"File {file_entity.id} metadata value '{value}' cannot be converted to new type '{new_value_type}', update skipped"
                                )
                                continue
                            value = converted_value

                        # Delete old key, add new key
                        file_entity.file_metadata.pop(old_name, None)
                        file_entity.file_metadata[new_name] = value
                        flag_modified(file_entity, "file_metadata")
                        self.session.add(file_entity)

            # If value_type changed, validate and convert all related files' values
            if old_value_type != new_value_type:
                logger.info(
                    f"Updating file's file_metadata value type: {old_value_type} -> {new_value_type}."
                )
                for file_entity in file_list:
                    if new_name in file_entity.file_metadata:
                        value = file_entity.file_metadata[new_name]
                        is_valid, converted_value = validate_metadata_value(
                            value, new_value_type
                        )
                        if is_valid:
                            flag_modified(file_entity, "file_metadata")
                            self.session.add(file_entity)
                        else:
                            logger.warning(
                                f"File {file_entity.id} metadata value '{value}' cannot be converted to new type '{new_value_type}', update skipped"
                            )

        # Flush to ensure changes are staged
        await self.session.flush()

        logger.info(
            f"Updated Metadata entity: {metadata_id} (name: {new_name}, updated {len(file_ids)} files)"
        )
        return updated_metadata


    async def delete_metadata(self, kb_id: str, metadata_id: str, tenant_id: str) -> None:
        """
        Delete metadata configuration and remove from all related files.
        This orchestrates the deletion across metadata, file metadata relation, and file services.

        Args:
            kb_id: Knowledgebase ID
            metadata_id: Metadata ID

        Raises:
            ValueError: If metadata not found
        """
        metadata_service = await self._get_metadata_service()
        metadata_entity = await metadata_service.get_metadata(kb_id=kb_id, metadata_id=metadata_id, tenant_id=tenant_id)
        if not metadata_entity:
            raise ValueError(f"Metadata '{metadata_id}' does not exist.")

        # Delete related file metadata relations
        file_metadata_relation_service = await self._get_file_metadata_relation_service()

        relations = await file_metadata_relation_service.get_file_metadata_relations_by_metadata_id(
            kb_id=kb_id, metadata_id=metadata_id, tenant_id=tenant_id
        )
        modified_file_count = 0

        if relations:
            file_ids = [relation.file_id for relation in relations]
            file_service = await self._get_file_service()
            for file_id in file_ids:
                file_entity = await file_service.get_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
                if file_entity and file_entity.kb_id == kb_id:
                    file_entity.file_metadata.pop(metadata_entity.name, None)
                    flag_modified(file_entity, "file_metadata")
                    self.session.add(file_entity)
                    modified_file_count += 1

        await file_metadata_relation_service.delete_file_metadata_relations_by_metadata_id(
            kb_id=kb_id, metadata_id=metadata_id, tenant_id=tenant_id
        )

        # Delete metadata entity
        await metadata_service.delete_metadata(kb_id=kb_id, metadata_id=metadata_id, tenant_id=tenant_id)

        # Flush to ensure all changes are staged
        await self.session.flush()

        logger.info(f"Deleted metadata {metadata_id} and all related file metadata relations (modified {modified_file_count} files)")


    async def set_file_metadata(
        self, kb_id: str, file_id: str, entry_data: MetadataEntryData, tenant_id: str
    ) -> KbFileEntity:
        """
        Set metadata for a file.
        This orchestrates the operation across metadata, file metadata relation, and file services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            entry_data: MetadataEntryData

        Returns:
            Updated KbFileEntity

        Raises:
            ValueError: If file not found or metadata validation fails
        """
        file_service = await self._get_file_service()
        metadata_service = await self._get_metadata_service()
        file_metadata_relation_service = await self._get_file_metadata_relation_service()

        # Validate file exists
        file_entity = await file_service.get_file(kb_id=kb_id, file_id=file_id, tenant_id=tenant_id)
        if not file_entity:
            raise ValueError(f"File '{file_id}' does not exist.")

        if file_entity.kb_id != kb_id:
            raise ValueError(f"File '{file_id}' does not belong to knowledgebase '{kb_id}'.")

        # Delete all existing FileMetadataEntity relations for this file
        await file_metadata_relation_service.delete_file_metadata_relations_by_file_id(
            kb_id=kb_id, file_id=file_id, tenant_id=tenant_id
        )

        # Update file_metadata JSON field
        file_metadata = {k: v for k, v in file_entity.file_metadata.items() if k in DEFAULT_METADATA_KEYS}
        for entry in entry_data.entries:
            metadata_name = entry.name
            metadata_value = entry.value

            if not metadata_name:
                continue

            # Get metadata entity to validate value type
            metadata_entity = await metadata_service.get_metadata_by_name(
                kb_id=kb_id, name=metadata_name, tenant_id=tenant_id
            )

            if metadata_entity:
                # Validate and convert value
                is_valid, converted_value = validate_metadata_value(
                    metadata_value, metadata_entity.value_type
                )
                if is_valid:
                    file_metadata[metadata_name] = converted_value

                    # Create FileMetadataEntity relation
                    await file_metadata_relation_service.create_file_metadata_relation(
                        kb_id=kb_id, file_id=file_id, metadata_id=metadata_entity.id, tenant_id=tenant_id
                    )
                else:
                    logger.warning(
                        f"Metadata value '{metadata_value}' cannot be converted to type '{metadata_entity.value_type}', update skipped"
                    )
            else:
                logger.warning(
                    f"Metadata '{metadata_name}' does not exist."
                )
        # Update file entity's file_metadata JSON
        file_entity.file_metadata = file_metadata
        flag_modified(file_entity, "file_metadata")
        self.session.add(file_entity)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(file_entity)

        logger.info(
            f"Set file metadata for file {file_id}: {len(entry_data.entries)} entries"
        )
        return file_entity


    async def format_search_result(self, reranked_result: VectorStoreQueryResult, tenant_id: str):
        records = []
        file_ids = []

        seen_file_urls = {}

        file_ids = set()
        for node in reranked_result.nodes:
            file_id = node.metadata.get("doc_id")
            if file_id:
                file_ids.add(file_id)

        file_source_map = await self.get_file_id_source_map(file_ids=file_ids, tenant_id=tenant_id)

        for i, node in enumerate(reranked_result.nodes):
            images = []
            origin_text = node.text
            pattern = MARKDOWN_IMAGE_PATTERN
            matches = re.findall(pattern, origin_text, re.DOTALL)
            for _, (src, desc)  in enumerate(matches):
                image_url = await file_store.get_url_async(file_path=src, tenant_id=tenant_id)
                origin_text = origin_text.replace(src, image_url)
                images.append({"url": image_url, "desc": desc})
            node.text = origin_text

            # TODO: Add file source
            file_url = file_source_map.get(node.metadata.get("doc_id"), None)
            file_path = node.metadata.get("file_path")
            node.metadata["file_source"] = file_url
            if not file_url and file_path:
                if file_path in seen_file_urls:
                    file_url = seen_file_urls[file_path]
                else:
                    file_url = await file_store.get_url_async(file_path=file_path, tenant_id=tenant_id)
                    seen_file_urls[file_path] = file_url


            records.append(
                SearchResult(
                    id=reranked_result.ids[i],
                    score=reranked_result.similarities[i],
                    content=origin_text,
                    images=images,
                    url=file_url,
                    title=node.metadata.get("file_name", ""),
                    metadata=node.metadata,
                ))
        return records


    @embedding_wrapper
    async def embed_query(self, query: str, embedding_model_entity: EmbeddingModelEntity) -> List[float]:
        embed_model = create_embedding_model(embedding_model_entity)
        # skip tracing for embedding vectors
        with suppress_tracing():
            query_embedding = await embed_model.aget_query_embedding(query)
            return query_embedding

    # 当需要发起SessionScope并发时，每个查询都需要独立的session实例
    async def _aquery_task(
        self,
        session: AsyncSession,
        kb_id: str = None,
        kb_name: str = None,
        query: str = None,
        user_id: str = None,
        retrieval_setting: Optional[RetrievalSetting] = None,
        metadata_condition: Optional[MetadataFilteringCondition] = None,
        document_ids: Optional[List[str]] = None,
        tenant_id: str = None,
    ) -> Tuple[Optional[VectorStoreQueryResult], Optional[VectorStoreQueryResult]]:
        knowledgebase_service = await self._get_kb_service(session)
        if kb_id:
            kb = await knowledgebase_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
            if not kb:
                raise ValueError(f"Knowledgebase id {kb_id} not found.")
        else:
            if kb_name:
                kb = await knowledgebase_service.get_knowledgebase_by_name(name=kb_name, tenant_id=tenant_id)
                if not kb:
                    raise ValueError(f"Knowledgebase name {kb_name} not found.")

                kb_id = kb.id
            else:
                raise ValueError("Knowledgebase ID or name is required.")

        embedding_service = await self._get_embedding_service(session)
        embed_model_entity = await embedding_service.get_embedding_model_by_provider_model_id(
            provider_name=kb.embedding_provider_name,
            model_id=kb.embedding_model,
            tenant_id=tenant_id,
        )
        if not embed_model_entity:
            raise ValueError(f"Embedding model not found for knowledgebase {kb_id}.")

        query_embedding = await self.embed_query(query=query, embedding_model_entity=embed_model_entity)

        if not retrieval_setting:
            retrieval_setting = RetrievalSetting()

        base_retrieval_setting = RetrievalConfig.model_validate(kb.retrieval_config)
        # save setting to dict
        if retrieval_setting.retrieval_mode is None:
            retrieval_setting.retrieval_mode = base_retrieval_setting.retrieval_mode
        if retrieval_setting.vector_weight is None:
            retrieval_setting.vector_weight = base_retrieval_setting.vector_weight
        if retrieval_setting.enable_rerank is None:
            retrieval_setting.enable_rerank = base_retrieval_setting.enable_rerank
        if retrieval_setting.rerank_model is None:
            retrieval_setting.rerank_model = base_retrieval_setting.rerank_model
        if retrieval_setting.rerank_provider_name is None:
            retrieval_setting.rerank_provider_name = base_retrieval_setting.rerank_provider_name
        if retrieval_setting.rerank_top_k is None:
            retrieval_setting.rerank_top_k = base_retrieval_setting.rerank_top_k
        if retrieval_setting.similarity_threshold is None:
            retrieval_setting.similarity_threshold = base_retrieval_setting.similarity_threshold
        if retrieval_setting.top_k is None:
            retrieval_setting.top_k = base_retrieval_setting.top_k

        query_mode = retrieval_type_to_search_mode(retrieval_setting.retrieval_mode)

        if not document_ids:
            try:
                document_ids = await query_file_ids_with_metadata_filter(session=session, kb_id=kb_id, user_id=user_id, metadata_filter=metadata_condition)
            except EmptyFilesException as e:
                logger.warning(f"{e}")
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])


        vector_db_service = await self._get_vector_db_service(session)
        vector_config = await vector_db_service.get_vectordb_config(tenant_id=tenant_id)
        if not vector_config:
            raise ValueError(f"VectorDB config not found for knowledgebase {kb_id}.")

        table_mapping_service = await self._get_vector_table_mapping_service(session)
        table_name = await table_mapping_service.get_vector_table_name(tenant_id=tenant_id, kb_id=kb.id)
        if not table_name:
            raise ValueError(f"Vector table name not found for knowledgebase {kb_id}.")
        vector_store = create_vector_store(
            kb_id=kb.id,
            dimension=len(query_embedding),
            vector_config=vector_config,
            table_name=table_name,
        )

        logger.info(f"Executing vector store query for query '{query}' against knowledgebase {kb_id} retrieval setting: {retrieval_setting}.")
        text_result, dense_result = await aquery_vector_store(
            kb_id=kb_id,
            vector_store=vector_store,
            query=query,
            query_embedding=query_embedding,
            document_ids=document_ids,
            query_mode=query_mode,
            top_k=retrieval_setting.top_k,
            use_docid_filter=is_docid_filter_supported(vector_store),
        )

        text_nodes_count = len(text_result.nodes) if text_result else 0
        dense_nodes_count = len(dense_result.nodes) if dense_result else 0

        logger.info(f"Retrieved {text_nodes_count} text nodes and {dense_nodes_count} dense nodes for query '{query}' against knowledgebase {kb_id}.")

        reranker = None
        if retrieval_setting.enable_rerank and retrieval_setting.rerank_model and (text_nodes_count + dense_nodes_count > 1):
            reranker_service = await self._get_reranker_service(session)
            reranker_config = await reranker_service.get_reranker_model_by_provider_model_id(
                provider_name=retrieval_setting.rerank_provider_name,
                model_id=retrieval_setting.rerank_model,
                tenant_id=tenant_id,
            )
            if not reranker_config:
                raise ValueError(f"Reranker model not found: provider {retrieval_setting.rerank_provider_name} and model {retrieval_setting.rerank_model}.")
            reranker = create_reranker_model(reranker_config)
            logger.info(f"Created reranker model {reranker_config.model_name} with provider {retrieval_setting.rerank_provider_name} and model {retrieval_setting.rerank_model}.")

        try:
            similarity_threshold = retrieval_setting.similarity_threshold or 0 # 如果没有设置similarity_threshold，直接返回所有结果

            reranked_result = await arerank_fusion(
                query=query,
                text_result=text_result,
                dense_result=dense_result,
                rerank_model=reranker,
                similarity_threshold=similarity_threshold,
                vector_weight=retrieval_setting.vector_weight or DEFAULT_VECTOR_WEIGHT,
                top_k=retrieval_setting.top_k or DEFAULT_SIMILARITY_TOP_K,
                rerank_top_k=retrieval_setting.rerank_top_k or DEFAULT_RERANK_SIMILARITY_TOP_K,
            )
            final_count = len(reranked_result.nodes) if reranked_result else 0
            if final_count < text_nodes_count + dense_nodes_count:
                logger.info(
                    f"After filtering (similarity_threshold={similarity_threshold}, rerank={retrieval_setting.enable_rerank}): "
                    f"{text_nodes_count + dense_nodes_count} -> {final_count} nodes for query '{query}'."
                )
            return reranked_result
        except Exception as e:
            logger.error(f"Failed to rerank: {e}")
            raise


    async def aquery_task(
        self,
        session: AsyncSession = None,
        kb_id: str = None,
        kb_name: str = None,
        query: str = None,
        user_id: str = None,
        retrieval_setting: Optional[RetrievalSetting] = None,
        metadata_condition: Optional[MetadataFilteringCondition] = None,
        document_ids: Optional[List[str]] = None,
        tenant_id: str = None,
    ):
        if session is None:
            async with create_db_session() as new_session:
                return await self._aquery_task(
                    session=new_session,
                    kb_id=kb_id,
                    kb_name=kb_name,
                    query=query,
                    user_id=user_id,
                    retrieval_setting=retrieval_setting,
                    metadata_condition=metadata_condition,
                    document_ids=document_ids,
                    tenant_id=tenant_id,
                )
        else:
            return await self._aquery_task(
                session=session,
                kb_id=kb_id,
                kb_name=kb_name,
                query=query,
                user_id=user_id,
                retrieval_setting=retrieval_setting,
                metadata_condition=metadata_condition,
                document_ids=document_ids,
                tenant_id=tenant_id,
            )

    ## VectorDB Retrieval Service
    @query_knowledgebase_wrapper
    async def aquery(
        self,
        query: str,
        kb_id: str = None,
        kb_name: str = None,
        kb_id_list: List[str] = None,
        user_id: str = None,
        tenant_id: str = None,
        retrieval_setting: Optional[RetrievalSetting] = None,
        metadata_condition: Optional[MetadataFilteringCondition] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        if not query:
            logger.info("No query provided, returning empty results.")
            return []


        if kb_id:
            reranked_result = await self.aquery_task(
                session=self.session,
                query=query,
                kb_id=kb_id,
                user_id=user_id,
                tenant_id=tenant_id,
                retrieval_setting=retrieval_setting,
                metadata_condition=metadata_condition,
                document_ids=document_ids,
            )
        elif kb_name:
            reranked_result = await self.aquery_task(
                session=self.session,
                query=query,
                kb_name=kb_name,
                user_id=user_id,
                tenant_id=tenant_id,
                retrieval_setting=retrieval_setting,
                metadata_condition=metadata_condition,
                document_ids=document_ids,
            )
        elif kb_id_list:
            if retrieval_setting.top_k and retrieval_setting.rerank_top_k is None:
                retrieval_setting.rerank_top_k = retrieval_setting.top_k

            vector_query_tasks = []
            for task_id in kb_id_list:
                task_retrieval_setting = copy.copy(retrieval_setting)

                vector_query_tasks.append(self.aquery_task(
                    kb_id=task_id,
                    user_id=user_id,
                    query=query,
                    tenant_id=tenant_id,
                    retrieval_setting=task_retrieval_setting,
                    metadata_condition=metadata_condition,
                    document_ids=document_ids,
                ))
            results = await asyncio.gather(*vector_query_tasks)
            reranked_result = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
            for i, sub_reranked_result in enumerate(results):
                if sub_reranked_result and sub_reranked_result.nodes:
                    logger.info(f"Retrieved {len(sub_reranked_result.nodes)} nodes from knowledgebase {kb_id_list[i]}")
                    reranked_result.nodes.extend(sub_reranked_result.nodes)
                    reranked_result.similarities.extend(sub_reranked_result.similarities)
                    reranked_result.ids.extend(sub_reranked_result.ids)

            # Truncate to top_k if results exceed limit
            top_k = retrieval_setting.top_k if retrieval_setting else DEFAULT_SIMILARITY_TOP_K
            if not top_k:
                top_k = DEFAULT_SIMILARITY_TOP_K

            # Sort by similarity descending and take top_k
            sorted_indices = sorted(
                range(len(reranked_result.similarities)),
                key=lambda i: reranked_result.similarities[i],
                reverse=True
            )[:top_k]
            reranked_result = VectorStoreQueryResult(
                nodes=[reranked_result.nodes[i] for i in sorted_indices],
                similarities=[reranked_result.similarities[i] for i in sorted_indices],
                ids=[reranked_result.ids[i] for i in sorted_indices],
            )
            logger.info(f"Truncated merged results from {len(reranked_result.nodes)} nodes to top_k={top_k}")
        else:
            raise ValueError("No valid knowledgebase ID or name provided.")

        records = await self.format_search_result(reranked_result, tenant_id)
        return records

    async def ainsert(
        self,
        kb_id: str,
        nodes: List[BaseNode],
        tenant_id: str,
    ) -> List[str]:
        logger.info(f"Starting to insert {len(nodes)} into vector store. Knowledgebase: {kb_id}")

        if not nodes:
            return []

        kb_service = await self._get_kb_service()
        kb = await kb_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
        if not kb:
            raise ValueError(f"Knowledgebase {kb_id} not found.")

        embed_service = await self._get_embedding_service()
        embed_model_entity = await embed_service.get_embedding_model_by_provider_model_id(
            provider_name=kb.embedding_provider_name,
            model_id=kb.embedding_model,
            tenant_id=tenant_id,
        )
        if not embed_model_entity:
            raise ValueError(f"Embedding model not found for knowledgebase {kb_id}.")

        embed_model = create_embedding_model(embed_model_entity)
        if hasattr(embed_model, "aget_multimodal_embedding_batch"):
            # 多模态向量模型：把节点中的图片直接送入 embedding，与文本融合为单个向量
            texts_to_embed = get_node_texts_for_embedding(nodes)
            mm_items = []
            for i, node in enumerate(nodes):
                images = []
                for img in (node.metadata or {}).get("images_info") or []:
                    url = img.get("url") if isinstance(img, dict) else None
                    if url:
                        images.append(url)
                mm_items.append({"text": texts_to_embed[i], "images": images})
            embeddings = await embed_model.aget_multimodal_embedding_batch(
                mm_items, show_progress=True
            )
        else:
            texts_to_embed = get_node_texts_for_embedding(nodes)
            embeddings = await embed_model.aget_text_embedding_batch(texts_to_embed, show_progress=True)
        for i in range(len(nodes)):
            nodes[i].embedding = embeddings[i]

        vector_db_service = await self._get_vector_db_service()
        vector_config = await vector_db_service.get_vectordb_config(tenant_id=tenant_id)
        if not vector_config:
            raise ValueError(f"VectorDB config not found for knowledgebase {kb_id}.")

        embed_dimension = len(embeddings[0])
        self._embed_dimension_cache.put(kb.embedding_model, embed_dimension)

        table_mapping_service = await self._get_vector_table_mapping_service()
        table_name = await table_mapping_service.get_vector_table_name(tenant_id=tenant_id, kb_id=kb.id)
        if not table_name:
            raise ValueError(f"Vector table name not found for knowledgebase {kb_id}.")
        vector_store = create_vector_store(
            kb_id=kb_id,
            dimension=embed_dimension,
            vector_config=vector_config,
            table_name=table_name,
        )

        try:
            node_ids = await vector_store.async_add(nodes)
            logger.info(f"Finished inserting {len(nodes)} into vector store. Node ids: {node_ids[:5]}")
            return node_ids
        except Exception as e:
            logger.error(f"Failed to insert nodes into vector store: {e}")
            raise


    async def adelete(
        self,
        kb_id: str,
        node_ids: List[str],
        tenant_id: str,
    ):
        if not node_ids:
            return []

        kb_service = await self._get_kb_service()
        kb = await kb_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
        if not kb:
            raise ValueError(f"Knowledgebase {kb_id} not found.")

        embed_dimension = self._embed_dimension_cache.get(kb.embedding_model)
        if not embed_dimension:
            embed_service = await self._get_embedding_service()
            embed_model_entity = await embed_service.get_embedding_model_by_provider_model_id(
                provider_name=kb.embedding_provider_name,
                model_id=kb.embedding_model,
                tenant_id=tenant_id,
            )
            if not embed_model_entity:
                raise ValueError(f"Embedding model not found for knowledgebase {kb_id}.")

            embed_model = create_embedding_model(embed_model_entity)
            embed_dimension = len(await embed_model.aget_text_embedding("0"))

        logger.info(f"Starting to delete {len(node_ids)} nodes from vector store. Node ids: {node_ids[:5]}...")
        vector_db_service = await self._get_vector_db_service()
        vector_config = await vector_db_service.get_vectordb_config(tenant_id=tenant_id)
        if not vector_config:
            raise ValueError(f"VectorDB config not found for knowledgebase {kb_id}.")

        table_mapping_service = await self._get_vector_table_mapping_service()
        table_name = await table_mapping_service.get_vector_table_name(tenant_id=tenant_id, kb_id=kb.id)
        if not table_name:
            raise ValueError(f"Vector table name not found for knowledgebase {kb_id}.")
        vector_store = create_vector_store(
            kb_id=kb_id,
            dimension=embed_dimension,
            vector_config=vector_config,
            table_name=table_name,
        )
        try:
            await vector_store.adelete_nodes(node_ids=node_ids)
            logger.info(f"Finished deleting {len(node_ids)} nodes from vector store. Node ids: {node_ids[:5]}...")
        except Exception as e:
            logger.error(f"Failed to delete nodes from vector store: {e}")
            raise

    async def delete_file(
        self,
        kb_id: str,
        file_id: str,
        tenant_id: str,
    ):
        kb_service = await self._get_kb_service()
        kb = await kb_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
        if not kb:
            raise ValueError(f"Knowledgebase {kb_id} not found.")

        embed_dimension = self._embed_dimension_cache.get(kb.embedding_model)
        if not embed_dimension:
            embed_service = await self._get_embedding_service()
            embed_model_entity = await embed_service.get_embedding_model_by_provider_model_id(
                provider_name=kb.embedding_provider_name,
                model_id=kb.embedding_model,
                tenant_id=tenant_id,
            )
            if not embed_model_entity:
                raise ValueError(f"Embedding model not found for knowledgebase {kb_id}.")

            embed_model = create_embedding_model(embed_model_entity)
            embed_dimension = len(await embed_model.aget_text_embedding("0"))

        logger.info(f"Starting to delete file {file_id} from vector store...")
        vector_db_service = await self._get_vector_db_service()
        vector_config = await vector_db_service.get_vectordb_config(tenant_id=tenant_id)
        if not vector_config:
            raise ValueError(f"VectorDB config not found for knowledgebase {kb_id}.")

        table_mapping_service = await self._get_vector_table_mapping_service()
        table_name = await table_mapping_service.get_vector_table_name(tenant_id=tenant_id, kb_id=kb.id)
        if not table_name:
            raise ValueError(f"Vector table name not found for knowledgebase {kb_id}.")
        vector_store = create_vector_store(
            kb_id=kb_id,
            dimension=embed_dimension,
            vector_config=vector_config,
            table_name=table_name,
        )
        try:
            await vector_store.adelete(ref_doc_id=file_id)
            logger.info(f"Finished deleting file {file_id} from vector store.")
        except Exception as e:
            logger.error(f"Failed to delete file {file_id} from vector store: {e}")
            raise

        chunk_service = await self._get_chunk_service()
        await chunk_service.delete_chunks_from_file(file_id=file_id, kb_id=kb_id, tenant_id=tenant_id)
        logger.info(f"Finished deleting chunks from file {file_id} from knowledgebase.")

        file_service = await self._get_file_service()
        await file_service.delete_file(file_id=file_id, kb_id=kb_id, tenant_id=tenant_id)
        logger.info(f"Finished deleting file {file_id} from knowledgebase.")

        # Also clean up the sync manifest so the next sync sees the document
        # as missing and re-ingests it. Without this, a manually deleted file
        # would be marked "unchanged" on the next sync and never re-fetched.
        ds_svc = DataSourceService(self.session)
        deleted_rows = await ds_svc.delete_document_rows_by_file_id(
            kb_id=kb_id, file_id=file_id, tenant_id=tenant_id
        )
        if deleted_rows:
            logger.info(
                f"Cleaned up {deleted_rows} sync manifest row(s) for "
                f"deleted file {file_id}"
            )
