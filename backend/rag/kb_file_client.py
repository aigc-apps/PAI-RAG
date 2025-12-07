import traceback
from typing import List, Optional
from db.models.knowledgebase.file_task import KbFileTaskEntity
from tqdm import tqdm
from db.models.knowledgebase.file import KbFileEntity
from db.models.knowledgebase.knowledgebase import KbEntity, ChunkConfig
from common.knowledgebase.types import (
    ChunkStatus,
    FileStatus,
)

from rag.offline_db_helper import (
    get_embedding_from_db,
    get_openailike_llm_from_db,
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
from rag.vector_store.vector_connection import cleanup_vector_store_async
from llama_index.core.embeddings import BaseEmbedding
from pairag.file.store.file_store_helper import file_store
from loguru import logger
from rag.parse_utils import sanitize_text, get_node_texts_for_embedding


class KbFileClient:

    async def create_file_parser(self, knowledgebase: KbEntity, image_caption_tool: Optional[ImageCaptionTool] = None):
        chunk_config = ChunkConfig.model_validate(knowledgebase.chunk_config)

        image_caption_tool = None
        if chunk_config.image_caption_model:
            multimodal_llm = await get_openailike_llm_from_db(model_id=chunk_config.image_caption_model)
            image_caption_tool = ImageCaptionTool(multimodal_llm=multimodal_llm)

        file_parser = FileParser(
            file_store=file_store,
            image_caption_tool=image_caption_tool,
            chunk_config=chunk_config,
        )
        return file_parser

    async def adelete_chunks_from_vectordb(
        self,
        kb_id: str,
        node_ids: List[str],
    ):
        if not node_ids:
            return

        knowledgebase: KbEntity = await get_knowledgebase_from_db(
            kb_id=kb_id
        )
        embed_model:BaseEmbedding = await get_embedding_from_db(model_id=knowledgebase.embedding_model)
        dimension = len(embed_model.get_text_embedding("0"))
        vector_store = await create_vector_store_from_db(kb_id=kb_id, dimension=dimension)
        await vector_store.adelete_nodes(node_ids=node_ids)
        logger.info(f"Deleted {len(node_ids)} chunks from {kb_id} vector db successfully.")


    # process file item, status -> processing
    # 这里是离线链路，所有的数据直接从db读取，不需要用到provider信息
    async def process_file_async(
        self,
        task_id: str,
        is_attachment: bool = False,
    ):
        logger.info(f"[WORKER] Start processing file task {task_id}. Is attachment: {is_attachment}")

        try:
            file_task: KbFileTaskEntity = await get_file_task_async(task_id=task_id)

            if not file_task:
                logger.warning(f"[WORKER] file task {task_id} not found. Process file task completed.")
                return

            if file_task.status == FileStatus.cancelled:
                logger.warning(f"[WORKER] file task {task_id} was cancelled. Process file task completed.")
                return

            file_id = file_task.file_id
            file_entity: KbFileEntity = await read_file_from_db(file_id=file_id)
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
                file = file_store.load(file_task.file_path)
                file_item = FileItem(
                    id=file_entity.id,
                    file_path=file_entity.file_path,
                    file=file,
                    kb_id=file_entity.kb_id,
                    file_extension=file_entity.file_extension,
                    file_name=file_entity.file_name,
                    file_md5=file_entity.file_md5,
                    file_size=file_entity.file_size,
                )

                kb_id = file_item.kb_id
                knowledgebase: KbEntity = await get_knowledgebase_from_db(
                    kb_id=kb_id
                )
                logger.info(
                    f"Start to add file {file_item.file_name} to knowledgebase {kb_id}."
                )
                await update_file_status_async(file_id=file_item.id, status=FileStatus.parsing, task_id=task_id, is_attachment=is_attachment)
            except Exception as ex:
                logger.error(f"处理文件失败：{traceback.format_exc()}")
                await update_file_status_async(file_id=file_id, status=FileStatus.failed, task_id=task_id, failed_reason=str(ex), is_attachment=is_attachment)


            if await should_cancel_file_task(
                file_id=file_id,
                kb_id=file_task.kb_id,
                file_part=file_task.file_part,
                file_version=file_task.file_version,
            ):
                return
            # parsing file
            logger.info(f"Parsing file {file_item.file_name}.")
            file_parser = await self.create_file_parser(knowledgebase)
            documents, nodes = file_parser.parse(file_item, is_attachment=is_attachment)
            await update_file_content_async(file_id=file_item.id, is_attachment=is_attachment,documents=documents)
            for node in nodes:
                # 去除\x00字符，适配postgresql
                node.text = sanitize_text(node.text)

            logger.info(f"Parsed {len(nodes)} documents.")

            if not nodes:
                logger.warning(f"No nodes parsed from file {file_item.file_name}. Marking file as completed.")
                await update_file_status_async(
                    file_id=file_item.id, task_id=task_id, status=FileStatus.succeeded, is_attachment=is_attachment
                )
                return

            # saving chunks
            if await should_cancel_file_task(
                file_id=file_id,
                kb_id=file_task.kb_id,
                file_part=file_task.file_part,
                file_version=file_task.file_version,
            ):
                return

            old_chunk_ids, new_chunk_ids = await save_chunks_to_db_async(
                kb_id=kb_id, file_id=file_item.id, file_part=file_task.file_part, chunk_nodes=nodes
            )
            logger.info(f"Saved {len(new_chunk_ids)} chunks to database.")

            # generate embedding
            if await should_cancel_file_task(
                file_id=file_id,
                kb_id=file_task.kb_id,
                file_part=file_task.file_part,
                file_version=file_task.file_version,
            ):
                return

            await update_file_status_async(
                file_id=file_item.id, task_id=task_id, status=FileStatus.persisting,
                is_attachment=is_attachment
            )
            logger.info(f"Starting to insert {len(nodes)} into knowledgebase {kb_id}.")
            embed_model:BaseEmbedding = await get_embedding_from_db(model_id=knowledgebase.embedding_model)

            dimension = len(embed_model.get_text_embedding("0"))
            vector_store = await create_vector_store_from_db(kb_id=kb_id, dimension=dimension)
            try:
                if old_chunk_ids:
                    try:
                        await vector_store.adelete_nodes(node_ids=old_chunk_ids)
                        logger.info(f"Removed {len(old_chunk_ids)} from vector store.")
                    except NotImplementedError:
                        logger.warning("Will not remove previous data as vector store does not support removing nodes.")
                        pass

                for i in tqdm(range(0, len(nodes), 1000), desc=f"Embedding & Persisting Nodes for file {file_item.file_name} part {file_task.file_part}"):
                    batch_nodes = nodes[i:i + 1000]
                    texts_to_embed = get_node_texts_for_embedding(batch_nodes)
                    embeddings = await embed_model.aget_text_embedding_batch(texts_to_embed, show_progress=False)
                    for j in range(len(batch_nodes)):
                        batch_nodes[j].embedding = embeddings[j]
                    if await should_cancel_file_task(
                        file_id=file_id,
                        kb_id=file_task.kb_id,
                        file_part=file_task.file_part,
                        file_version=file_task.file_version,
                    ):
                        # 在返回前清理连接，避免连接泄漏
                        await cleanup_vector_store_async(vector_store)
                        return
                    await vector_store.async_add(batch_nodes)
            finally:
                # 确保无论成功还是失败都清理连接，避免连接泄漏
                await cleanup_vector_store_async(vector_store)
            logger.info(f"Finished inserting {len(nodes)} into knowledgebase {kb_id}.")
            await update_chunk_status_async(chunk_ids=new_chunk_ids, status=ChunkStatus.succeeded)
            await update_file_status_async(
                file_id=file_item.id, task_id=task_id, status=FileStatus.succeeded, is_attachment=is_attachment
            )
            logger.info(
                f"Finished adding file {file_item.file_name} to knowledgebase {kb_id}."
            )
        except Exception as e:
            await update_file_status_async(
                file_id=file_item.id, task_id=task_id, status=FileStatus.failed, is_attachment=is_attachment, failed_reason=str(e),
            )
            logger.error(f"Error processing file: {traceback.format_exc()}")


kb_file_client = KbFileClient()
