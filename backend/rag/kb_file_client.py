import traceback
from typing import List, Optional
from db.models.knowledgebase.file_task import KbFileTaskEntity
from llama_index.core.schema import BaseNode
import asyncio
from tqdm import tqdm
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.knowledgebase import KbEntity
from pairag.file.nodeparsers.file_parser import ChunkConfig, TableParserConfig
from common.knowledgebase.types import (
    ChunkStatus,
    FileStatus,
)

from rag.offline_db_helper import (
    get_embedding_from_db,
    get_llm_model_from_db,
    get_file_task_async,
    read_file_from_db,
    save_chunks_to_db_async,
    update_chunk_status_async,
    update_file_status_async,
    update_file_content_async,
    should_cancel_file_task,
    get_knowledgebase_from_db,
    create_vector_store_from_db,
)
from pairag.file.models.file_item import FileItem
from pairag.file.nodeparsers.file_parser import FileParser
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from pairag.file.utils.video_caption_tool import VideoCaptionTool
from pairag.file.utils.multimodal_llm import OpenAIMultimodalLLM
from rag.vector_store.vector_connection import cleanup_vector_store_async
from llama_index.core.embeddings import BaseEmbedding
from pairag.file.store.file_store_helper import file_store
from common.encrypt_utils import decrypt_key
from loguru import logger
from rag.parse_utils import sanitize_text, get_node_texts_for_embedding
from common.knowledgebase.constants import DEFAULT_PARAGRAPH_SEPARATOR
from rag.convert.office_converter import convert_doc_to_docx, convert_ppt_to_pptx

def convert_file_if_needed(file_item: FileItem):
    if file_item.file_extension == ".ppt":
        file_item.file = convert_ppt_to_pptx(file_item.file)
    elif file_item.file_extension == ".doc":
        file_item.file = convert_doc_to_docx(file_item.file)

MAX_CONCURRENT_PERSIST_TASK_COUNT = 10

class KbFileClient:

    async def create_file_parser(self, knowledgebase: KbEntity, file_entity: Optional[KbFileEntity] = None, image_caption_tool: Optional[ImageCaptionTool] = None):
        # Prioritize file's chunk_config if available, otherwise use knowledgebase's chunk_config
        if file_entity and file_entity.chunk_config:
            chunk_config = ChunkConfig.model_validate(file_entity.chunk_config)
        else:
            chunk_config = ChunkConfig.model_validate(knowledgebase.chunk_config)

        image_caption_tool = None
        video_caption_tool = None
        if chunk_config.image_caption_model:
            llm_config = await get_llm_model_from_db(
                model_id=chunk_config.image_caption_model,
                tenant_id=knowledgebase.tenant_id,
                provider_name=chunk_config.image_caption_provider_name,
            )
            if not llm_config:
                logger.error(f"Image caption model {chunk_config.image_caption_model} not found.")
                raise ValueError(f"Image caption model {chunk_config.image_caption_model} not found.")

            multimodal_llm = OpenAIMultimodalLLM(base_url=llm_config.base_url, api_key=decrypt_key(llm_config.encrypted_api_key), model=llm_config.model)
            image_caption_tool = ImageCaptionTool(multimodal_llm=multimodal_llm)
            video_caption_tool = VideoCaptionTool(multimodal_llm=multimodal_llm)

        if not chunk_config.separator:
            chunk_config.separator = DEFAULT_PARAGRAPH_SEPARATOR

        # Ensure table_config is not None
        if chunk_config.table_config is None:
            chunk_config.table_config = TableParserConfig()

        file_parser = FileParser(
            file_store=file_store,
            image_caption_tool=image_caption_tool,
            video_caption_tool=video_caption_tool,
            chunk_config=chunk_config,
        )
        return file_parser

    async def adelete_chunks_from_vectordb(
        self,
        kb_id: str,
        node_ids: List[str],
        tenant_id: str = None,
    ):
        if not node_ids:
            return

        knowledgebase: KbEntity = await get_knowledgebase_from_db(
            kb_id=kb_id,
            tenant_id=tenant_id,
        )
        embed_model:BaseEmbedding = await get_embedding_from_db(model_id=knowledgebase.embedding_model, tenant_id=tenant_id, provider_name=knowledgebase.embedding_provider_name)
        dimension = len(await embed_model.aget_text_embedding("0"))
        vector_store = await create_vector_store_from_db(kb_id=kb_id, dimension=dimension, tenant_id=tenant_id)
        try:
            await vector_store.adelete_nodes(node_ids=node_ids)
            logger.info(f"Deleted {len(node_ids)} chunks from {kb_id} vector db successfully.")
        finally:
            await cleanup_vector_store_async(vector_store)


    # process file item, status -> processing
    # 这里是离线链路，所有的数据直接从db读取，不需要用到provider信息
    async def process_file_async(
        self,
        task_id: str,
        is_attachment: bool = False,
        tenant_id: str = None,
    ):
        logger.info(f"[WORKER] Start processing file task {task_id} for tenant {tenant_id}. Is attachment: {is_attachment}")

        try:
            file_task: KbFileTaskEntity = await get_file_task_async(task_id=task_id, tenant_id=tenant_id)

            if not file_task:
                logger.warning(f"[WORKER] file task {task_id} not found. Process file task completed.")
                return

            if file_task.status == FileStatus.cancelled:
                logger.warning(f"[WORKER] file task {task_id} was cancelled. Process file task completed.")
                return

            file_id = file_task.file_id
            file_entity: KbFileEntity = await read_file_from_db(file_id=file_id, tenant_id=tenant_id)
            if not file_entity:
                logger.warning(f"[WORKER] file {file_id} not found. Process file task completed.")
                return

            if file_entity.status == FileStatus.cancelled:
                logger.warning(f"[WORKER] file {file_id} was cancelled. Process file task completed.")
                return

            if file_entity.file_version != file_task.file_version:
                logger.warning(f"[WORKER] file {file_id} was updated. Process file task completed.")
                return


            try:
                file = await file_store.read_async(file_path=file_task.file_path, tenant_id=tenant_id)
                file_item = FileItem(
                    id=file_entity.id,
                    file_path=file_entity.file_path,
                    file=file,
                    kb_id=file_entity.kb_id,
                    file_extension=file_entity.file_extension,
                    file_name=file_entity.file_name,
                    file_md5=file_entity.file_md5,
                    file_size=file_entity.file_size,
                    tenant_id=tenant_id,
                )

                kb_id = file_item.kb_id
                knowledgebase: KbEntity = await get_knowledgebase_from_db(
                    kb_id=kb_id,
                    tenant_id=tenant_id,
                )
                logger.info(
                    f"Start to add file {file_item.file_name} to knowledgebase {kb_id}."
                )
                await update_file_status_async(file_id=file_item.id, status=FileStatus.parsing, task_id=task_id, is_attachment=is_attachment, tenant_id=tenant_id)
            except Exception as ex:
                logger.error(f"Fail to process file: {traceback.format_exc()}")
                await update_file_status_async(file_id=file_id, status=FileStatus.failed, task_id=task_id, failed_reason=str(ex), is_attachment=is_attachment, tenant_id=tenant_id)
                return

            if await should_cancel_file_task(
                file_id=file_id,
                kb_id=file_task.kb_id,
                file_part=file_task.file_part,
                file_version=file_task.file_version,
                tenant_id=tenant_id,
            ):
                return
            # parsing file
            logger.info(f"Parsing file {file_item.file_name}.")
            file_parser = await self.create_file_parser(knowledgebase, file_entity=file_entity)
            convert_file_if_needed(file_item)
            documents, nodes = file_parser.parse(file_item, is_attachment=is_attachment)
            await update_file_content_async(file_id=file_item.id, is_attachment=is_attachment, documents=documents, tenant_id=tenant_id)

            # Prepend the document title (e.g. from Yuque) directly into each
            # chunk's text content so it participates in embedding naturally.
            # The MarkdownNodeParser already prepends the section heading
            # hierarchy to the chunk text; we add the document-level title on
            # top of that so the full context is in the chunk content itself.
            file_meta = file_entity.file_metadata or {}
            file_title = file_meta.get('title', '').strip()
            if file_title:
                for node in nodes:
                    existing_title = node.metadata.get('title', '').strip()
                    if existing_title and existing_title != file_title:
                        node.metadata['title'] = f"{file_title} > {existing_title}"
                    else:
                        node.metadata['title'] = file_title
                    # Prepend document title into chunk content directly
                    node.text = f"# {file_title}\n\n{node.text}"

            for node in nodes:
                # 去除\x00字符，适配postgresql
                node.text = sanitize_text(node.text)

            logger.info(f"Parsed {len(nodes)} documents.")

            if not nodes:
                logger.warning(f"No nodes parsed from file {file_item.file_name}. Marking file as completed.")
                await update_file_status_async(
                    file_id=file_item.id, task_id=task_id, status=FileStatus.succeeded, is_attachment=is_attachment, tenant_id=tenant_id
                )
                return

            # saving chunks
            if await should_cancel_file_task(
                file_id=file_id,
                kb_id=file_task.kb_id,
                file_part=file_task.file_part,
                file_version=file_task.file_version,
                tenant_id=tenant_id,
            ):
                return

            old_chunk_ids, new_chunk_ids = await save_chunks_to_db_async(
                kb_id=kb_id, file_id=file_item.id, file_part=file_task.file_part, chunk_nodes=nodes, tenant_id=tenant_id
            )
            logger.info(f"Saved {len(new_chunk_ids)} chunks to database.")

            # generate embedding
            if await should_cancel_file_task(
                file_id=file_id,
                kb_id=file_task.kb_id,
                file_part=file_task.file_part,
                file_version=file_task.file_version,
                tenant_id=tenant_id,
            ):
                return

            await update_file_status_async(
                file_id=file_item.id, task_id=task_id, status=FileStatus.persisting,
                is_attachment=is_attachment,
                tenant_id=tenant_id,
            )
            logger.info(f"Starting to insert {len(nodes)} into knowledgebase {kb_id}.")
            embed_model:BaseEmbedding = await get_embedding_from_db(model_id=knowledgebase.embedding_model, tenant_id=tenant_id, provider_name=knowledgebase.embedding_provider_name)

            dimension = len(await embed_model.aget_text_embedding("0"))
            vector_store = await create_vector_store_from_db(kb_id=kb_id, dimension=dimension, tenant_id=tenant_id)
            persist_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PERSIST_TASK_COUNT)
            async def _persist_nodes_async(nodes: List[BaseNode], pbar: tqdm):
                logger.info(f"Trying to persist {len(nodes)} nodes to vector store.")
                async with persist_semaphore:
                    logger.info(f"Persisting {len(nodes)} nodes to vector store.")
                    await vector_store.async_add(nodes)
                    logger.info(f"Persisted {len(nodes)} nodes to vector store.")
                    pbar.update(len(nodes))

            try:
                if old_chunk_ids:
                    try:
                        await vector_store.adelete_nodes(node_ids=old_chunk_ids)
                        logger.info(f"Removed {len(old_chunk_ids)} from vector store.")
                    except NotImplementedError:
                        logger.warning("Will not remove previous data as vector store does not support removing nodes.")
                        pass

                embed_batch_size = embed_model.embed_batch_size
                if embed_batch_size <= 0:
                    embed_batch_size = 10

                persist_tasks = []
                persist_progress_bar = tqdm(total=len(nodes), desc=f"Embedding & Persisting Nodes for file {file_item.file_name} part {file_task.file_part}")
                n_batches = (len(nodes) - 1) // embed_batch_size + 1

                supports_multimodal = hasattr(embed_model, "aget_multimodal_embedding_batch")
                for i in range(0, len(nodes), embed_batch_size):
                    batch_nodes = nodes[i:i + embed_batch_size]
                    texts_to_embed = get_node_texts_for_embedding(batch_nodes)
                    if supports_multimodal:
                        # 多模态向量模型：把节点中的图片直接送入 embedding，与文本融合为单个向量
                        mm_items = []
                        for k, node in enumerate(batch_nodes):
                            images = []
                            for img in (node.metadata or {}).get("images_info") or []:
                                url = img.get("url") if isinstance(img, dict) else None
                                if url:
                                    images.append(url)
                            mm_items.append({"text": texts_to_embed[k], "images": images})
                        embeddings = await embed_model.aget_multimodal_embedding_batch(
                            mm_items, show_progress=False
                        )
                    else:
                        embeddings = await embed_model.aget_text_embedding_batch(texts_to_embed, show_progress=False)
                    # 在 for 循环里，append 之后加一行，例如：
                    persist_progress_bar.set_postfix_str(f"embedded {(i // embed_batch_size) + 1}/{n_batches} batches")
                    for j in range(len(batch_nodes)):
                        batch_nodes[j].embedding = embeddings[j]
                    if await should_cancel_file_task(
                        file_id=file_id,
                        kb_id=file_task.kb_id,
                        file_part=file_task.file_part,
                        file_version=file_task.file_version,
                        tenant_id=tenant_id,
                    ):
                        for t in persist_tasks:
                            t.cancel()
                        if persist_tasks:
                            await asyncio.gather(*persist_tasks, return_exceptions=True)
                        await cleanup_vector_store_async(vector_store)
                        return
                    # create_task 立即把协程挂到 event loop，embedding 下一批时 persist 已在后台跑
                    persist_tasks.append(asyncio.create_task(_persist_nodes_async(batch_nodes, persist_progress_bar)))

                if persist_tasks:
                    results = await asyncio.gather(*persist_tasks, return_exceptions=True)
                    persist_errors = [r for r in results if isinstance(r, Exception)]
                    if persist_errors:
                        logger.error(f"Failed to persist {len(persist_errors)} batches to vector store: {persist_errors}")
                        raise persist_errors[0]
                persist_progress_bar.close()
            finally:
                # 确保无论成功还是失败都清理连接，避免连接泄漏
                await cleanup_vector_store_async(vector_store)
            logger.info(f"Finished inserting {len(nodes)} into knowledgebase {kb_id}.")
            await update_chunk_status_async(chunk_ids=new_chunk_ids, status=ChunkStatus.succeeded, tenant_id=tenant_id)
            await update_file_status_async(
                file_id=file_item.id, task_id=task_id, status=FileStatus.succeeded, is_attachment=is_attachment, tenant_id=tenant_id
            )
            logger.info(
                f"Finished adding file {file_item.file_name} to knowledgebase {kb_id}."
            )
        except Exception as e:
            error_file_id = file_item.id if 'file_item' in dir() else (file_id if 'file_id' in dir() else None)
            if error_file_id:
                await update_file_status_async(
                    file_id=error_file_id, task_id=task_id, status=FileStatus.failed, is_attachment=is_attachment, failed_reason=str(e), tenant_id=tenant_id,
                )
            logger.error(f"Error processing file: {traceback.format_exc()}")
            raise


kb_file_client = KbFileClient()
