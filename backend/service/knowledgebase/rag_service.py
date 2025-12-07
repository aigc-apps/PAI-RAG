"""RAG Service layer for orchestrating business logic across knowledgebase entities."""

from typing import Callable, Awaitable, Optional, List
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

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
from db.models.knowledgebase.chunk import KbChunkEntity, create_text_node_from_chunk
from service.knowledgebase.utils.metadata_utils import validate_metadata_value
from service.factory.model_factory import create_reranker_model
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
        kb_service_getter: Callable[[], Awaitable],
        file_service_getter: Callable[[], Awaitable],
        chunk_service_getter: Callable[[], Awaitable],
        metadata_service_getter: Callable[[], Awaitable],
        file_metadata_relation_service_getter: Callable[[], Awaitable],
        embedding_service_getter: Callable[[], Awaitable],
        reranker_service_getter: Callable[[], Awaitable],
        llm_service_getter: Callable[[], Awaitable],
        vector_db_service_getter: Callable[[], Awaitable],
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

        self._embed_dimension_cache = LruCache(max_size=100)

    # ------------------------------------------------------------------------------------------------
    # Knowledgebase operations: list, add, get, update, delete
    # ------------------------------------------------------------------------------------------------

    async def list_knowledgebases(
        self,
        page: int = 1,
        size: int = 10,
        query: Optional[str] = None,
        exclude_default_attachments: bool = True,
    ) -> PagedResult[List[KbEntity]]:
        kb_service = await self._get_kb_service()
        return await kb_service.list_knowledgebases(page, size, query, exclude_default_attachments)


    async def _validate_knowledgebase_models(
        self, knowledgebase: KnowledgebaseCreate
    ) -> None:
        """
        Validate that all referenced models (embedding, reranker, image_caption) exist and are valid.

        Args:
            knowledgebase: Knowledgebase data to validate

        Raises:
            ValueError: If any model is invalid or doesn't exist
        """
        # Validate embedding_model
        logger.info(f"Validating knowledgebase models: {knowledgebase}")
        if not knowledgebase.embedding_model:
            raise ValueError("需要提供嵌入模型才能创建知识库。")

        embedding_service = await self._get_embedding_service()
        embedding_model = await embedding_service.get_embedding_by_model_id(
            knowledgebase.embedding_model
        )
        if not embedding_model:
            raise ValueError(
                f"嵌入模型 '{knowledgebase.embedding_model}' 不存在。"
            )

        # Validate rerank_model if rerank is enabled
        if knowledgebase.retrieval_config:
            retrieval_config = knowledgebase.retrieval_config
            if retrieval_config.enable_rerank:
                if not retrieval_config.rerank_model:
                    raise ValueError("启用重排序时，必须指定重排序模型。")
                reranker_service = await self._get_reranker_service()
                reranker_model = await reranker_service.get_reranker_by_model_id(
                    retrieval_config.rerank_model
                )
                if not reranker_model:
                    raise ValueError(
                        f"重排序模型 '{retrieval_config.rerank_model}' 不存在。"
                    )

        # Validate image_caption_model if specified
        if knowledgebase.chunk_config and knowledgebase.chunk_config.image_caption_model:
            llm_service = await self._get_llm_service()
            llm_model = await llm_service.get_llm_by_model_id(
                knowledgebase.chunk_config.image_caption_model
            )
            if not llm_model:
                raise ValueError(
                    f"图片描述模型 '{knowledgebase.chunk_config.image_caption_model}' 不存在。"
                )
            if not llm_model.vision_support:
                raise ValueError(
                    f"图片描述模型 '{knowledgebase.chunk_config.image_caption_model}' 不支持视觉功能。"
                )

    async def add_knowledgebase(
        self, knowledgebase: KnowledgebaseCreate
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
        await self._validate_knowledgebase_models(knowledgebase)

        # Create knowledgebase
        kb_service = await self._get_kb_service()
        return await kb_service.create_knowledgebase(knowledgebase)


    async def update_knowledgebase(
        self, kb_id: str, knowledgebase: KnowledgebaseCreate
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
        await self._validate_knowledgebase_models(knowledgebase)

        # Update knowledgebase
        return await kb_service.update_knowledgebase(kb_id, knowledgebase)

    async def get_knowledgebase_by_name(self, name: str) -> Optional[KbEntity]:
        """
        Get a knowledgebase by name.
        This orchestrates the retrieval across knowledgebase services.

        Args:
            name: Knowledgebase name
        """
        kb_service = await self._get_kb_service()
        return await kb_service.get_knowledgebase_by_name(name)

    async def get_knowledgebase(self, kb_id: str) -> Optional[KbEntity]:
        """
        Get a knowledgebase.
        This orchestrates the retrieval across knowledgebase, file, chunk, metadata, and file metadata relation services.

        Args:
            kb_id: Knowledgebase ID

        Returns:
            KbEntity if found, None otherwise
        """
        kb_service = await self._get_kb_service()
        return await kb_service.get_knowledgebase(kb_id)

    async def delete_knowledgebase(self, kb_id: str) -> None:
        """
        Delete a knowledgebase and all related entities (files, chunks, metadata).
        This orchestrates the deletion across multiple services.

        Args:
            kb_id: Knowledgebase ID

        Raises:
            ValueError: If knowledgebase not found
        """
        kb_service = await self._get_kb_service()
        knowledgebase = await kb_service.get_knowledgebase(kb_id)
        if not knowledgebase:
            raise ValueError(f"知识库 '{kb_id}' 不存在。")

        # Delete related files directly (no need to query first)
        file_service = await self._get_file_service()
        await file_service.delete_files_from_kb(kb_id)

        # Delete related chunks directly (no need to query first)
        chunk_service = await self._get_chunk_service()
        chunk_ids = await chunk_service.delete_chunks_from_kb(kb_id)
        await self.adelete(kb_id=kb_id, node_ids=chunk_ids)

        # Delete related metadata in batch
        metadata_service = await self._get_metadata_service()
        await metadata_service.delete_metadata_by_kb_id(kb_id)

        # Delete related file metadata relations
        file_metadata_relation_service = await self._get_file_metadata_relation_service()
        await file_metadata_relation_service.delete_file_metadata_relations_by_kb_id(kb_id)

        # Delete knowledgebase itself
        await kb_service.delete_knowledgebase(kb_id)

        logger.info(f"Deleted knowledgebase {kb_id} and all related entities")


    # ------------------------------------------------------------------------------------------------
    # File operations: get, add, update, delete
    # ------------------------------------------------------------------------------------------------

    async def list_files(
        self,
        kb_id: str,
        page: int = 1,
        size: int = 10,
        query: Optional[str] = None,
        status: Optional[str] = None,
    ) -> PagedResult[List[KbFileEntity]]:
        """
        List files in a knowledgebase.

        Args:
            kb_id: Knowledgebase ID
            page: Page number (1-indexed)
            size: Page size
            query: Optional search query (searches in file_name)
            status: Optional filter for file status

        Returns:
            PagedResult containing list of KbFileEntity and pagination metadata
        """
        file_service = await self._get_file_service()
        return await file_service.list_files(kb_id, page, size, query, status)


    async def get_files_by_path(self, kb_id: str, file_paths: List[str]) -> List[KbFileEntity]:
        """
        Get a file by path from a knowledgebase.
        This orchestrates the retrieval across file services.

        Args:
            kb_id: Knowledgebase ID
            file_path: File path
        """
        file_service = await self._get_file_service()
        return await file_service.get_files_by_path(kb_id, file_paths)


    async def get_file_by_name(self, kb_id: str, file_name: str) -> Optional[KbFileEntity]:
        """
        Get a file by name from a knowledgebase.
        This orchestrates the retrieval across file services.

        Args:
            kb_id: Knowledgebase ID
            file_name: File name
        """
        file_service = await self._get_file_service()
        return await file_service.get_file_by_name(kb_id, file_name)


    async def get_file(self, kb_id: str, file_id: str) -> KbFileEntity:
        """
        Get a file from a knowledgebase.
        This orchestrates the retrieval across file and chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
        """
        file_service = await self._get_file_service()
        return await file_service.get_file(kb_id, file_id)


    async def add_file(self, kb_id: str, file: KbFileEntity) -> KbFileEntity:
        """
        Add a file to a knowledgebase.
        This orchestrates the addition across file and chunk services.

        Args:
            kb_id: Knowledgebase ID
            file: File data
        """
        file_service = await self._get_file_service()
        return await file_service.add_file(kb_id, file)


    async def delete_file(self, kb_id: str, file_id: str) -> None:
        """
        Delete a file and all related chunks.
        This orchestrates the deletion across file and chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID

        Raises:
            ValueError: If file not found
        """
        file_service = await self._get_file_service()
        file_entity = await file_service.get_file(kb_id, file_id)
        if not file_entity:
            raise ValueError(f"文件 '{file_id}' 不存在。")

        if file_entity.kb_id != kb_id:
            raise ValueError(f"文件 '{file_id}' 不属于知识库 '{kb_id}'。")

        # Delete related chunks in batch
        chunk_service = await self._get_chunk_service()
        chunk_ids = await chunk_service.delete_chunks_from_file(file_id, kb_id)
        await self.adelete(kb_id=kb_id, node_ids=chunk_ids)

        # Delete file itself
        await file_service.delete_file(file_id, kb_id)

        logger.info(f"Deleted file {file_id} and all related chunks")


    ## Batch
    async def batch_delete_files(self, kb_id: str, file_ids: List[str]) -> None:
        """
        Batch delete files and related chunks.
        This orchestrates the deletion across file and chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_ids: List of file IDs
        """
        file_service = await self._get_file_service()
        for file_id in file_ids:
            await self.delete_file(kb_id, file_id)
        return await file_service.batch_delete_files(kb_id, file_ids)


    async def get_file_id_source_map(
        self,
        kb_id: str,
        file_ids: List[str],
    ):
        file_source_results = (await self.session.exec(
            select( KbFileEntity.id, KbFileEntity.file_source ).where(
                KbFileEntity.kb_id == kb_id,
                KbFileEntity.id.in_(file_ids),
            )
        )).all()
        logger.info(f"Get file_source_results: {file_source_results}")
        return {
            file_id: file_source
            for file_id, file_source in file_source_results
        }

    # Chunk operations: get, add, update, delete, list

    async def list_chunks(self, kb_id: str, file_id: str, page: int = 1, size: int = 10) -> PagedResult[List[KbChunkEntity]]:
        """
        List chunks in a file.
        This orchestrates the retrieval across chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
        """
        chunk_service = await self._get_chunk_service()
        return await chunk_service.list_chunks(kb_id, file_id, page, size)


    async def get_chunk(self, kb_id: str, file_id: str, chunk_id: str) -> KbChunkEntity:
        """
        Get a chunk from a file.
        This orchestrates the retrieval across chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            chunk_id: Chunk ID
        """
        chunk_service = await self._get_chunk_service()
        return await chunk_service.get_chunk(kb_id, file_id, chunk_id)


    async def add_chunk(self, kb_id: str, file_id: str, text: str, chunk_metadata: dict = None) -> KbChunkEntity:
        """
        Add a chunk to a file.
        This orchestrates the addition across chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            text: Chunk text
            chunk_metadata: Chunk metadata
        """
        file_entity = await self.get_file(kb_id, file_id)
        if not file_entity:
            raise ValueError(f"文件 '{file_id}' 不存在。")

        chunk_metadata = chunk_metadata or file_entity.file_metadata

        chunk_service = await self._get_chunk_service()
        chunk = await chunk_service.create_chunk(kb_id, file_id, text, chunk_metadata)

        kb_node = create_text_node_from_chunk(chunk)
        await self.ainsert(kb_id=kb_id, nodes=[kb_node])
        return chunk


    async def update_chunk(self, kb_id: str, file_id: str, chunk_id: str, chunk: KbChunkEntity) -> KbChunkEntity:
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
        kb_chunk = await chunk_service.update_chunk(kb_id=kb_id, file_id=file_id, chunk_id=chunk_id, new_chunk=chunk)
        kb_node = create_text_node_from_chunk(kb_chunk)

        await self.adelete(kb_id=kb_id, node_ids=[kb_chunk.id])
        if kb_chunk.active:
            await self.ainsert(kb_id=kb_id, nodes=[kb_node])
        return kb_chunk


    async def delete_chunk(self, kb_id: str, file_id: str, chunk_id: str) -> None:
        """
        Delete a chunk from a file.
        This orchestrates the deletion across chunk services.

        Args:
            kb_id: Knowledgebase ID
            file_id: File ID
            chunk_id: Chunk ID
        """
        await self.adelete(kb_id=kb_id, node_ids=[chunk_id])

        chunk_service = await self._get_chunk_service()
        await chunk_service.delete_chunk(kb_id=kb_id, file_id=file_id, chunk_id=chunk_id)

    # ------------------------------------------------------------------------------------------------
    # Metadata operations: list, get, add, update, delete
    # ------------------------------------------------------------------------------------------------
    async def list_metadata(self, kb_id: str, page: int = 1, size: int = 10) -> PagedResult[List[KbMetadataEntity]]:
        """
        List metadata in a knowledgebase.
        This orchestrates the retrieval across metadata services.

        Args:
            kb_id: Knowledgebase ID
        """
        metadata_service = await self._get_metadata_service()
        return await metadata_service.list_metadata(kb_id, page, size)

    async def get_metadata(self, kb_id: str, metadata_id: str) -> Optional[KbMetadataEntity]:
        """
        Get a metadata from a knowledgebase.
        This orchestrates the retrieval across metadata services.

        Args:
            kb_id: Knowledgebase ID
            metadata_id: Metadata ID
        """
        metadata_service = await self._get_metadata_service()
        return await metadata_service.get_metadata(kb_id, metadata_id)

    async def update_metadata(
        self,
        kb_id: str,
        metadata_id: str,
        update_data: KbMetadataEntityCreate,
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
        metadata_entity = await metadata_service.get_metadata(kb_id, metadata_id)
        if not metadata_entity:
            raise ValueError(f"元数据 '{metadata_id}' 不存在。")

        old_name = metadata_entity.name
        old_value_type = metadata_entity.value_type
        new_name = update_data.name
        new_value_type = update_data.value_type

        # Update metadata entity itself
        updated_metadata = await metadata_service.update_metadata(
            kb_id, metadata_id, update_data
        )

        # Find all files using this metadata
        file_metadata_relation_service = await self._get_file_metadata_relation_service()
        file_metadata_list = await file_metadata_relation_service.get_file_metadata_relations_by_metadata_id(
            kb_id, metadata_id
        )
        file_ids = [relation.file_id for relation in file_metadata_list]
        if file_ids:
            file_service = await self._get_file_service()
            file_list = await file_service.get_files_by_ids(kb_id, file_ids)
            # If name changed, update all related files' file_metadata key names
            if old_name != new_name:
                logger.info(
                    f"更新文件的 file_metadata 中的键名: {old_name} -> {new_name}."
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
                                    f"文件 {file_entity.id} 的元数据值 '{value}' 无法转换为新类型 '{new_value_type}'，跳过更新"
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
                    f"更新文件的 file_metadata 中的值类型: {old_value_type} -> {new_value_type}."
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
                                f"文件 {file_entity.id} 的元数据值 '{value}' 无法转换为新类型 '{new_value_type}'，跳过更新"
                            )

        # Flush to ensure changes are staged
        await self.session.flush()

        logger.info(
            f"Updated Metadata entity: {metadata_id} (name: {new_name}, updated {len(file_ids)} files)"
        )
        return updated_metadata


    async def delete_metadata(self, kb_id: str, metadata_id: str) -> None:
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
        metadata_entity = await metadata_service.get_metadata(kb_id, metadata_id)
        if not metadata_entity:
            raise ValueError(f"元数据 '{metadata_id}' 不存在。")

        # Delete related file metadata relations
        file_metadata_relation_service = await self._get_file_metadata_relation_service()

        relations = await file_metadata_relation_service.get_file_metadata_relations_by_metadata_id(
            kb_id, metadata_id
        )
        if relations:
            file_ids = [relation.file_id for relation in relations]
            file_service = await self._get_file_service()
            modified_file_count = 0
            for file_id in file_ids:
                file_entity = await file_service.get_file(kb_id, file_id)
                if file_entity and file_entity.kb_id == kb_id:
                    file_entity.file_metadata.pop(metadata_entity.name, None)
                    flag_modified(file_entity, "file_metadata")
                    self.session.add(file_entity)
                    modified_file_count += 1

        await file_metadata_relation_service.delete_file_metadata_relations_by_metadata_id(
            kb_id, metadata_id
        )

        # Delete metadata entity
        await metadata_service.delete_metadata(kb_id, metadata_id)

        # Flush to ensure all changes are staged
        await self.session.flush()

        logger.info(f"Deleted metadata {metadata_id} and all related file metadata relations (modified {modified_file_count} files)")


    async def set_file_metadata(
        self, kb_id: str, file_id: str, entry_data: MetadataEntryData
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
        file_entity = await file_service.get_file(kb_id, file_id)
        if not file_entity:
            raise ValueError(f"文件 '{file_id}' 不存在。")

        if file_entity.kb_id != kb_id:
            raise ValueError(f"文件 '{file_id}' 不属于知识库 '{kb_id}'。")

        # Delete all existing FileMetadataEntity relations for this file
        await file_metadata_relation_service.delete_file_metadata_relations_by_file_id(
            kb_id, file_id
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
                kb_id, metadata_name
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
                        kb_id, file_id, metadata_entity.id
                    )
                else:
                    logger.warning(
                        f"元数据值 '{metadata_value}' 无法转换为类型 '{metadata_entity.value_type}'，跳过"
                    )
            else:
                logger.warning(
                    f"元数据 '{metadata_name}' 不存在。"
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


    ## VectorDB Retrieval Service
    async def aquery(
        self,
        query: str,
        knowledge_id: str = None,
        knowledge_name: str = None,
        user_id: str = None,
        retrieval_setting: Optional[RetrievalSetting] = None,
        metadata_condition: Optional[MetadataFilteringCondition] = None,
        document_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        knowledgebase_service = await self._get_kb_service()
        if knowledge_id:
            kb = await knowledgebase_service.get_knowledgebase(knowledge_id)
        else:
            if knowledge_name:
                kb = await knowledgebase_service.get_knowledgebase_by_name(knowledge_name)
            else:
                raise ValueError("Knowledgebase ID or name is required.")

        if not kb:
            raise ValueError(f"Knowledgebase {knowledge_id} not found.")

        embedding_service = await self._get_embedding_service()
        embed_model = await embedding_service.get_embedding_model(kb.embedding_model)
        if not embed_model:
            raise ValueError(f"Embedding model not found for knowledgebase {knowledge_id}.")

        base_retrieval_setting = RetrievalConfig.model_validate(kb.retrieval_config)
        if not retrieval_setting:
            retrieval_setting = base_retrieval_setting
        else:
            if retrieval_setting.retrieval_mode is None:
                retrieval_setting.retrieval_mode = base_retrieval_setting.retrieval_mode
            if retrieval_setting.vector_weight is None:
                retrieval_setting.vector_weight = base_retrieval_setting.vector_weight
            if retrieval_setting.enable_rerank is None:
                retrieval_setting.enable_rerank = base_retrieval_setting.enable_rerank
            if retrieval_setting.rerank_model is None:
                retrieval_setting.rerank_model = base_retrieval_setting.rerank_model
            if retrieval_setting.rerank_top_k is None:
                retrieval_setting.rerank_top_k = base_retrieval_setting.rerank_top_k
            if retrieval_setting.similarity_threshold is None:
                retrieval_setting.similarity_threshold = base_retrieval_setting.similarity_threshold
            if retrieval_setting.top_k is None:
                retrieval_setting.top_k = base_retrieval_setting.top_k

        if not document_ids:
            try:
                document_ids = await query_file_ids_with_metadata_filter(session=self.session, kb_id=knowledge_id, user_id=user_id, metadata_filter=metadata_condition)
            except EmptyFilesException as e:
                logger.error(f"No files found with the given metadata filter. error: {e}")
                return []

        query_embedding = await embed_model.aget_query_embedding(query)
        query_mode = retrieval_type_to_search_mode(retrieval_setting.retrieval_mode)


        vector_db_service = await self._get_vector_db_service()
        vector_config = await vector_db_service.get_vectordb_config()
        if not vector_config:
            raise ValueError(f"VectorDB config not found for knowledgebase {knowledge_id}.")

        vector_store = create_vector_store(
            kb_id=kb.id,
            dimension=len(query_embedding),
            vector_config=vector_config,
        )

        logger.info("Executing vector store query...")

        text_result, dense_result = await aquery_vector_store(
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
        logger.info(f"Executing rerank phrase...text nodes: {text_nodes_count}, dense nodes: {dense_nodes_count}")
        reranker = None
        if retrieval_setting.enable_rerank and retrieval_setting.rerank_model and (text_nodes_count + dense_nodes_count > 1):
            reranker_service = await self._get_reranker_service()
            reranker_config = await reranker_service.get_reranker_by_model_id(
                retrieval_setting.rerank_model
            )
            if not reranker_config:
                raise ValueError(f"Reranker model not found for knowledgebase {knowledge_id}.")
            reranker = create_reranker_model(reranker_config)
            logger.info(f"Created reranker model {reranker_config.model_name} for knowledgebase {knowledge_id}.")

        try:
            reranked_result = await arerank_fusion(
                query=query,
                text_result=text_result,
                dense_result=dense_result,
                rerank_model=reranker,
                vector_weight=retrieval_setting.vector_weight,
                top_k=retrieval_setting.top_k,
            )
        except Exception as e:
            logger.error(f"Failed to rerank: {e}")
            raise

        records = []
        file_ids = []
        for i, node in enumerate(reranked_result.nodes):
            if reranked_result.similarities[i] >= retrieval_setting.similarity_threshold:
                images = []
                file_ids.append(node.metadata["doc_id"])
                origin_text = node.text
                pattern = MARKDOWN_IMAGE_PATTERN
                matches = re.findall(pattern, origin_text, re.DOTALL)
                for _, (src, desc)  in enumerate(matches):
                    image_url = file_store.get_url(src)
                    origin_text = origin_text.replace(src, image_url)
                    images.append({"url": image_url, "desc": desc})
                node.text = origin_text

                # TODO: Add file source
                file_url = node.metadata.get("file_source")
                if not file_url:
                    file_url = file_store.get_url(node.metadata.get("file_path", ""))
                records.append(
                    SearchResult(
                        score=reranked_result.similarities[i],
                        content=origin_text[:3000],
                        images=images,
                        url=file_url,
                        title=node.metadata.get("file_name", ""),
                        metadata=node.metadata,
                    ).model_dump())

        # TODO add file source map
        logger.info(f"Get {len(records)} nodes above given threshold {retrieval_setting.similarity_threshold}.")
        return records

    async def ainsert(
        self,
        kb_id: str,
        nodes: List[BaseNode],
    ) -> List[str]:
        logger.info(f"Starting to insert {len(nodes)} into vector store. Knowledgebase: {kb_id}")

        if not nodes:
            return []

        kb_service = await self._get_kb_service()
        kb = await kb_service.get_knowledgebase(kb_id)
        if not kb:
            raise ValueError(f"Knowledgebase {kb_id} not found.")

        embed_service = await self._get_embedding_service()
        embed_model = await embed_service.get_embedding_model(kb.embedding_model)
        if not embed_model:
            raise ValueError(f"Embedding model not found for knowledgebase {kb_id}.")

        texts_to_embed = get_node_texts_for_embedding(nodes)
        embeddings = await embed_model.aget_text_embedding_batch(texts_to_embed, show_progress=True)
        for i in range(len(nodes)):
            nodes[i].embedding = embeddings[i]

        vector_db_service = await self._get_vector_db_service()
        vector_config = await vector_db_service.get_vectordb_config()
        if not vector_config:
            raise ValueError(f"VectorDB config not found for knowledgebase {kb_id}.")

        embed_dimension = len(embeddings[0])
        self._embed_dimension_cache.put(kb.embedding_model, embed_dimension)
        vector_store = create_vector_store(
            kb_id=kb_id,
            dimension=embed_dimension,
            vector_config=vector_config,
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
    ):
        if not node_ids:
            return []

        kb_service = await self._get_kb_service()
        kb = await kb_service.get_knowledgebase(kb_id)
        if not kb:
            raise ValueError(f"Knowledgebase {kb_id} not found.")

        embed_dimension = self._embed_dimension_cache.get(kb.embedding_model)
        if not embed_dimension:
            embed_serivce = await self._get_embedding_service()
            embed_model = await embed_serivce.get_embedding_model(kb.embedding_model)
            if not embed_model:
                raise ValueError(f"Embedding model not found for knowledgebase {kb_id}.")
            embed_dimension = len(await embed_model.aget_text_embedding("0"))

        logger.info(f"Starting to delete {len(node_ids)} nodes from vector store. Node ids: {node_ids[:5]}...")
        vector_db_service = await self._get_vector_db_service()
        vector_config = await vector_db_service.get_vectordb_config()
        if not vector_config:
            raise ValueError(f"VectorDB config not found for knowledgebase {kb_id}.")
        vector_store = create_vector_store(
            kb_id=kb_id,
            dimension=embed_dimension,
            vector_config=vector_config,
        )
        try:
            await vector_store.adelete_nodes(node_ids=node_ids)
            logger.info(f"Finished deleting {len(node_ids)} nodes from vector store. Node ids: {node_ids[:5]}...")
        except Exception as e:
            logger.error(f"Failed to delete nodes from vector store: {e}")
            raise

    async def adelete_file(
        self,
        kb_id: str,
        file_id: str,
    ):
        kb_service = await self._get_kb_service()
        kb = await kb_service.get_knowledgebase(kb_id)
        if not kb:
            raise ValueError(f"Knowledgebase {kb_id} not found.")

        embed_dimension = self._embed_dimension_cache.get(kb.embedding_model)
        if not embed_dimension:
            embed_service = await self._get_embedding_service()
            embed_model = await embed_service.get_embedding_model(kb.embedding_model)
            if not embed_model:
                raise ValueError(f"Embedding model not found for knowledgebase {kb_id}.")
            embed_dimension = len(await embed_model.aget_text_embedding("0"))

        logger.info(f"Starting to delete file {file_id} from vector store...")
        vector_db_service = await self._get_vector_db_service()
        vector_config = await vector_db_service.get_vectordb_config()
        if not vector_config:
            raise ValueError(f"VectorDB config not found for knowledgebase {kb_id}.")
        vector_store = create_vector_store(
            kb_id=kb_id,
            dimension=embed_dimension,
            vector_config=vector_config,
        )
        try:
            await vector_store.adelete(ref_doc_id=file_id)
            logger.info(f"Finished deleting file {file_id} from vector store.")
        except Exception as e:
            logger.error(f"Failed to delete file {file_id} from vector store: {e}")
            raise
