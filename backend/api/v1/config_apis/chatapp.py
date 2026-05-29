### Embedding configuration API ###

import time
import asyncio
import tempfile
import os
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Query, File, UploadFile, Form
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.chatbot import (
    ChatBotCreate,
    ChatBotEntity,
)
from pairag.file.models.file_item import FileItem
import uuid
import hashlib
from rag.kb_file_client import kb_file_client
from rag.parse_utils import sanitize_text
from tqdm import tqdm
from db.db_context import get_db_session
from sqlalchemy.exc import IntegrityError
from common.chat.response_model import PagedResult, ResponseModel, success_response
from service.injection import get_chatapp_service, get_tenant_id, get_faq_config_service, get_faq_item_service, get_rag_service, get_embedding_service, get_knowledgebase_service, get_file_service
from service.tool.chatapp_service import ChatappService
from service.tool.faq_config_service import FAQConfigService
from service.tool.faq_item_service import FAQItemService
from service.injection import get_faq_item_service
from db.models.faq_item import FAQItemCreate
from service.knowledgebase.rag_service import RagService
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from service.knowledgebase.file_service import FileService
from service.model.embedding_service import EmbeddingService
from db.models.knowledgebase.knowledgebase import KnowledgebaseCreate, RetrievalConfig
from pairag.file.nodeparsers.file_parser import ChunkConfig, TableParserConfig
from common.knowledgebase.constants import FAQ_KNOWLEDGEBASE_NAME, DEFAULT_FAQ_SIMILARITY_THRESHOLD
from common.knowledgebase.types import VectorIndexRetrievalType, FileStatus
from rag.file_item_utils import to_file_entity
from typing import Optional, List
from io import BytesIO
import json
from api.api_exception import ApiException, handle_api_exceptions
import traceback
from loguru import logger
from common.i18n import i18n

app_router = APIRouter()

# Import FAQ dependencies
from db.models.faq_config import FAQConfigCreate
from db.models.faq_item import FAQItemCreate, FAQItemEntity

# FAQ routes - MUST be defined before /{id} routes to avoid route conflicts
# FastAPI matches routes in order, so more specific routes must come first
@app_router.post("/{app_id}/faqs", response_model=ResponseModel[FAQItemEntity], tags=["FAQ"])
@handle_api_exceptions(action="create FAQ item", i18n_error_key="api.faq.create_failed", default_code=500)
async def create_faq_item(
    app_id: str,
    faq_item_create: FAQItemCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    faq_item_service: FAQItemService = Depends(get_faq_item_service),
    rag_service: RagService = Depends(get_rag_service),
):
    logger.info(f"Creating FAQ item for app_id: {app_id}")
    # Get chatbot by app_id to get chatbot_id
    chatbot = await chatapp_service.get_chatapp_by_app_id(app_id=app_id, tenant_id=tenant_id)
    if not chatbot:
        raise ApiException(code=404, message=f"Chat App '{app_id}' does not exist.")

    if not chatbot.enable_faq or not chatbot.faq_config:
        raise ApiException(code=400, message="FAQ is not enabled.")

    faq_item = await faq_item_service.create_faq_item(
        chatbot_id=chatbot.app_id,
        faq_item_data=faq_item_create,
        tenant_id=tenant_id,
    )
    await faq_item_service.save_faq_to_knowledgebase(faq_item, tenant_id, rag_service)
    await session.commit()
    await session.refresh(faq_item)
    return success_response(data=faq_item, message=i18n.t("api.faq.create_success"))

@app_router.get("/{app_id}/faqs", tags=["FAQ"])
@handle_api_exceptions(action="list FAQ items", i18n_error_key="api.faq.list_failed", default_code=500)
async def list_faq_items(
    app_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=100, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    faq_item_service: FAQItemService = Depends(get_faq_item_service),
):
    logger.info(f"Listing FAQ items for app_id: {app_id}")
    # Get chatbot by app_id to get chatbot_id
    chatbot = await chatapp_service.get_chatapp_by_app_id(app_id=app_id, tenant_id=tenant_id)
    if not chatbot:
        raise ApiException(code=404, message=f"Chat app '{app_id}' does not exist.")

    faq_items = await faq_item_service.list_faq_items(
        chatbot_id=chatbot.app_id,
        tenant_id=tenant_id,
        page=page,
        size=size,
    )
    return success_response(data=faq_items, message=i18n.t("api.faq.list_success"))

@app_router.put("/{app_id}/faqs/{faq_item_id}", response_model=ResponseModel[FAQItemEntity], tags=["FAQ"])
@handle_api_exceptions(action="update FAQ item", i18n_error_key="api.faq.update_failed", default_code=500)
async def update_faq_item(
    app_id: str,
    faq_item_id: str,
    faq_item_update: FAQItemCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    faq_item_service: FAQItemService = Depends(get_faq_item_service),
    rag_service: RagService = Depends(get_rag_service),
):
    faq_item = await faq_item_service.update_faq_item(
        id=faq_item_id, update_data=faq_item_update, tenant_id=tenant_id, rag_service=rag_service
    )
    await session.commit()
    await session.refresh(faq_item)
    return success_response(data=faq_item, message=i18n.t("api.faq.update_success"))

@app_router.delete("/{app_id}/faqs/{faq_item_id}", tags=["FAQ"])
@handle_api_exceptions(action="delete FAQ item", i18n_error_key="api.faq.delete_failed", default_code=500)
async def delete_faq_item(
    app_id: str,
    faq_item_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    faq_item_service: FAQItemService = Depends(get_faq_item_service),
    rag_service: RagService = Depends(get_rag_service),
):
    await faq_item_service.delete_faq_item(id=faq_item_id, tenant_id=tenant_id, rag_service=rag_service)
    await session.commit()
    return success_response(message=i18n.t("api.faq.delete_success", id=faq_item_id))

MAX_CHECK_ATTEMPTS = 100
CHECK_INTERVAL = 3

@app_router.post("/{app_id}/faq-files", tags=["FAQ"])
async def upload_faq_files(
    app_id: str,
    files: Optional[List[UploadFile]] = File(...),
    table_config: Optional[str] = Form(None, description="JSON string of table_config, shared by all files in this upload"),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    knowledgebase_service: KnowledgebaseService = Depends(get_knowledgebase_service),
    rag_service: RagService = Depends(get_rag_service),
    tenant_id: str = Depends(get_tenant_id),
):
    """Upload and parse FAQ files directly without storing file entity."""
    knowledgebase = None
    try:
        if not files:
            raise ApiException(code=400, message=i18n.t("api.error.no_files"))
        # Get chatbot by app_id to get chatbot_id
        chatbot = await chatapp_service.get_chatapp_by_app_id(app_id=app_id, tenant_id=tenant_id)
        if not chatbot:
            raise ApiException(code=404, message=i18n.t("api.app.not_found", id=app_id))

        if not chatbot.enable_faq or not chatbot.faq_config:
            raise ApiException(code=400, message=i18n.t("api.faq.enable_first"))

        # Get or create FAQ knowledgebase
        kb_name = f"{app_id}_{FAQ_KNOWLEDGEBASE_NAME}"
        knowledgebase = await knowledgebase_service.get_knowledgebase_by_name(kb_name, tenant_id=tenant_id)
        default_embedding_config = await embedding_service.get_default_embedding(tenant_id=tenant_id)
        kb_id = None

        if not knowledgebase:
            logger.info(f"Creating FAQ knowledgebase {kb_name} for tenant {tenant_id}")
            faq_config_service = FAQConfigService(session)
            faq_config = await faq_config_service.get_or_create_faq_config(
                chatbot=chatbot
            )

            if faq_config and faq_config.embedding_model:
                embedding_model = faq_config.embedding_model
            else:
                embedding_model = default_embedding_config.model_id

            default_similarity_threshold = faq_config.similarity_threshold if faq_config else DEFAULT_FAQ_SIMILARITY_THRESHOLD

            retrieval_config = RetrievalConfig(
                retrieval_mode=VectorIndexRetrievalType.vector,
                top_k=1,
                enable_rerank=False,
                rerank_top_k=None,
                vector_weight=1.0,
                similarity_threshold=default_similarity_threshold,
            )

            kb_create = KnowledgebaseCreate(
                name=kb_name,
                description=i18n.t("api.faq.kb_description"),
                embedding_model=embedding_model,
                retrieval_config=retrieval_config,
            )
            knowledgebase = await knowledgebase_service.create_knowledgebase(kb_data=kb_create, tenant_id=tenant_id)
            kb_id = knowledgebase.id
            try:
                await session.commit()
                await session.refresh(knowledgebase)
                await knowledgebase_service.write_cache_after_commit(knowledgebase, tenant_id)
                logger.info(f"Created FAQ knowledgebase {kb_id} for tenant {tenant_id}")
            except IntegrityError:
                await session.rollback()
                await knowledgebase_service.delete_cache_on_rollback(kb_id, tenant_id, kb_create.name)
                knowledgebase = await knowledgebase_service.get_knowledgebase_by_name(kb_name, tenant_id=tenant_id)
                if not knowledgebase:
                    raise ApiException(code=500, message=i18n.t("api.faq.kb_create_failed"))
                kb_id = knowledgebase.id
                logger.info(f"Retrieved existing FAQ knowledgebase {kb_id} for tenant {tenant_id}")
            except Exception:
                await session.rollback()
                await knowledgebase_service.delete_cache_on_rollback(kb_id, tenant_id, kb_create.name)
                raise
        else:
            kb_id = knowledgebase.id
            logger.info(f"Found existing FAQ knowledgebase {kb_id} for tenant {tenant_id}")



        parsed_chunk_config = knowledgebase.chunk_config
        parsed_chunk_config = ChunkConfig.model_validate(parsed_chunk_config)

        # Parse and validate table_config if provided
        parsed_table_config = None
        if table_config:
            try:
                table_config_dict = json.loads(table_config)
                if not isinstance(table_config_dict, dict):
                    raise ValueError("table_config must be a JSON object")
                parsed_table_config = TableParserConfig.model_validate(table_config_dict)
                if parsed_chunk_config.table_config:
                    table_config_dict_merged = parsed_chunk_config.table_config.model_dump()
                    table_config_dict_merged.update(table_config_dict)
                    parsed_chunk_config.table_config = TableParserConfig.model_validate(table_config_dict_merged)
                else:
                    parsed_chunk_config.table_config = parsed_table_config
            except json.JSONDecodeError as e:
                raise ApiException(code=400, message=i18n.t("api.knowledgebase.table_config_error", error=str(e)))
            except Exception as e:
                raise ApiException(code=400, message=i18n.t("api.knowledgebase.table_config_validation_failed", error=str(e)))



        # Process each file
        response_data = []
        total_chunks = 0

        for file in files:
            temp_file_path = None
            try:
                file_content = await file.read()
                file_content_io = BytesIO(file_content)
                file_extension = "." + (file.filename.split(".")[-1] if "." in file.filename else "")

                # Create temporary file to store file content
                temp_file = tempfile.NamedTemporaryFile(
                    mode='wb',
                    suffix=file_extension,
                    delete=False,
                    prefix=f"faq_upload_{uuid.uuid4().hex}_"
                )
                temp_file_path = temp_file.name
                temp_file.write(file_content)
                temp_file.close()

                # Create FileItem for parsing
                file_id = uuid.uuid4().hex
                file_md5 = hashlib.md5(file_content).hexdigest()

                file_item = FileItem(
                    id=file_id,
                    file_path=temp_file_path,
                    file=file_content_io,
                    kb_id=knowledgebase.id,
                    file_extension=file_extension,
                    file_name=file.filename or f"faq_file_{file_id}",
                    file_md5=file_md5,
                    file_size=len(file_content),
                    tenant_id=tenant_id,
                )

                logger.info(f"Parsing FAQ file {file_item.file_name} from temporary path {temp_file_path}...")
                file_entity = to_file_entity(file_item=file_item)
                # Convert ChunkConfig object to dict for file_entity
                file_entity.chunk_config = parsed_chunk_config.model_dump()
                file_parser = await kb_file_client.create_file_parser(knowledgebase, file_entity)
                documents, nodes = file_parser.parse(file_item, is_attachment=False)

                if not nodes:
                    logger.warning(f"No nodes parsed from file {file_item.file_name}.")
                    response_data.append({"file_name": file_item.file_name, "items_count": 0})
                    continue

                # Sanitize text
                for node in nodes:
                    node.text = sanitize_text(node.text)

                logger.info(f"Parsed {len(nodes)} documents from FAQ file {file_item.file_name}.")

                # Save FAQ items to database from node metadata
                faq_item_service = FAQItemService(session)
                saved_faq_count = 0

                # Prepare FAQ items data from all nodes
                faq_items_to_create = []
                for node in nodes:
                    question = node.metadata.get("question", "").strip() if node.metadata else ""
                    answer = node.metadata.get("answer", "").strip() if node.metadata else ""

                    # Skip if both question and answer are empty
                    if not question and not answer:
                        logger.warning(f"Skipping node with empty question and answer from file {file_item.file_name}")
                        continue

                    faq_item_data = FAQItemCreate(
                        question=question,
                        answer=answer,
                        chatbot_id=chatbot.app_id,
                        file_id=file_id,
                        active=True,
                    )
                    faq_items_to_create.append(faq_item_data)

                if faq_items_to_create:
                    created_faq_items = []
                    for faq_item_data in tqdm(faq_items_to_create, desc=f"Creating FAQ Items for file {file_item.file_name}"):
                        try:
                            faq_item = await faq_item_service.create_faq_item(
                                chatbot_id=chatbot.app_id,
                                faq_item_data=faq_item_data,
                                tenant_id=tenant_id,
                            )
                            created_faq_items.append(faq_item)
                        except Exception as create_error:
                            logger.error(f"Failed to create FAQ item: {create_error}")
                            continue

                    # Save to knowledgebase in parallel batches
                    if created_faq_items:
                        save_tasks = [
                            faq_item_service.save_faq_to_knowledgebase(faq_item, tenant_id, rag_service)
                            for faq_item in created_faq_items
                        ]

                        if save_tasks:
                            # Process save tasks in batches to avoid overwhelming the system
                            save_batch_size = 50
                            for j in tqdm(range(0, len(save_tasks), save_batch_size), desc=f"Saving FAQ Items to KB for file {file_item.file_name}"):
                                save_batch = save_tasks[j:j + save_batch_size]
                                await asyncio.gather(*save_batch, return_exceptions=True)

                        saved_faq_count = len(created_faq_items)

                        # Commit all FAQ items
                        await session.commit()

                        # Refresh all created items (skip if refresh fails)
                        for faq_item in created_faq_items:
                            try:
                                await session.refresh(faq_item)
                            except Exception as refresh_error:
                                logger.debug(f"Could not refresh FAQ item {faq_item.id} (may not be persistent): {refresh_error}")

                logger.info(f"Saved {saved_faq_count}/{len(nodes)} FAQ items to database from file {file_item.file_name}.")

                # Add successful result to response_data
                response_data.append({
                    "file_name": file_item.file_name,
                    "items_count": saved_faq_count
                })
                total_chunks += saved_faq_count
            except Exception as file_error:
                logger.error(f"Failed to process FAQ file {file.filename}: {traceback.format_exc()}")
                response_data.append({
                    "file_name": file.filename,
                    "items_count": 0,
                    "error": str(file_error)
                })
                # Continue processing other files even if one fails
            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                        logger.debug(f"Deleted temporary file: {temp_file_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to delete temporary file {temp_file_path}: {cleanup_error}")

        await session.commit()

        logger.info(f"Uploaded {len(files)} FAQ files successfully, total chunks: {total_chunks}.")
        return success_response(
            data=response_data,
            message=i18n.t("api.faq.upload_success", count=len(files), chunks=total_chunks)
        )
    except Exception as e:
        logger.error(f"Failed to process FAQ file: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.faq.file_process_failed", error=str(e)))


@app_router.post("", response_model=ResponseModel[ChatBotEntity])
@handle_api_exceptions(action="create chatapp", i18n_error_key="api.app.create_failed", default_code=500)
async def create_chatbot(
    chatbot_create: ChatBotCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
):
    chatbot = await chatapp_service.create_chatapp(app_data=chatbot_create, tenant_id=tenant_id)
    await session.commit()
    await session.refresh(chatbot)
    return success_response(data=chatbot, message=i18n.t("api.app.create_success"))


@app_router.get("")
@handle_api_exceptions(action="list chatapps", i18n_error_key="api.app.list_failed", default_code=500)
async def get_chatbots(
    app_id: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
):
    if not app_id:
        chatbots = await chatapp_service.list_chatapps(page=page, size=size, tenant_id=tenant_id)
        return success_response(data=chatbots, message=i18n.t("api.app.list_success"))
    chatbot = await chatapp_service.get_chatapp_by_app_id(app_id=app_id, tenant_id=tenant_id)
    if not chatbot:
        raise ApiException(code=404, message=i18n.t("api.app.query_failed", id=app_id))
    return success_response(data=chatbot, message=i18n.t("api.app.query_success"))


@app_router.put("/{id}", response_model=ResponseModel[ChatBotEntity])
@handle_api_exceptions(action="update chatapp", i18n_error_key="api.app.update_failed", default_code=500)
async def update_chatbot(
    id: str,
    new_chatbot: ChatBotCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
):
    chatbot = await chatapp_service.update_chatapp(id=id, update_data=new_chatbot, tenant_id=tenant_id)
    return success_response(data=chatbot, message=i18n.t("api.app.update_success"))


@app_router.delete("/{id}")
@handle_api_exceptions(action="delete chatapp", i18n_error_key="api.app.delete_failed", default_code=500)
async def delete_chatbot(
    id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    rag_service: RagService = Depends(get_rag_service),
):
    await chatapp_service.delete_chatapp(id=id, tenant_id=tenant_id, rag_service=rag_service)
    await session.commit()
    return success_response(message=i18n.t("api.app.delete_success", id=id))
