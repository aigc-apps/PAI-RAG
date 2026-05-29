### Knowledgebase configuration API ###
from typing import List
from fastapi import Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.knowledgebase.file import KbFileEntity, MetadataEntryData
from db.db_context import get_db_session
from db.models.knowledgebase.metadata import KbMetadataEntity, KbMetadataEntityCreate
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import handle_api_exceptions
from api.v1.config_apis.knowledgebase import knowledgebase_router
from service.knowledgebase.rag_service import RagService
from service.knowledgebase.metadata_service import MetadataService
from service.injection import get_rag_service, get_metadata_service, get_tenant_id



@knowledgebase_router.post("/{kb_id}/metadata", response_model=ResponseModel[KbMetadataEntity])
@handle_api_exceptions(action="create metadata")
async def add_kb_metadata(
    kb_id: str,
    metadata_create: KbMetadataEntityCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
    metadata_service: MetadataService = Depends(get_metadata_service),
):
    metadata_entity = await metadata_service.create_metadata(kb_id=kb_id, metadata_create=metadata_create, tenant_id=tenant_id)
    return success_response(data=metadata_entity, message="Metadata created successfully.")


@knowledgebase_router.get("/{kb_id}/metadata", response_model=ResponseModel[List[dict]])
@handle_api_exceptions(action="list metadata")
async def list_metadata(
    kb_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    metadata_list = await rag_service.list_metadata(kb_id=kb_id, tenant_id=tenant_id, page=page, size=size)
    return success_response(data=metadata_list, message="List metadata success.")


@knowledgebase_router.put("/{kb_id}/metadata/{metadata_id}", response_model=ResponseModel[List[KbMetadataEntity]])
@handle_api_exceptions(action="update metadata")
async def update_metadata(
    kb_id: str,
    metadata_id: str,
    new_metadata_entity: KbMetadataEntity,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    metadata_entity = await rag_service.update_metadata(
        kb_id=kb_id,
        metadata_id=metadata_id,
        update_data=new_metadata_entity,
        tenant_id=tenant_id)
    return success_response(data=metadata_entity, message="Update metadata success.")


@knowledgebase_router.delete("/{kb_id}/metadata/{metadata_id}", response_model=ResponseModel[KbMetadataEntity])
@handle_api_exceptions(action="delete metadata")
async def delete_metadata(
    kb_id: str,
    metadata_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    metadata_entity = await rag_service.delete_metadata(kb_id=kb_id, metadata_id=metadata_id, tenant_id=tenant_id)
    return success_response(data=metadata_entity, message="Delete metadata success.")


@knowledgebase_router.post("/{kb_id}/files/{file_id}/metadata", response_model=ResponseModel[KbFileEntity])
@handle_api_exceptions(action="set file metadata")
async def set_file_metadata(
    kb_id: str,
    file_id: str,
    entry_data: MetadataEntryData,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    rag_service: RagService = Depends(get_rag_service),
):
    file_entity = await rag_service.set_file_metadata(kb_id=kb_id, file_id=file_id, entry_data=entry_data, tenant_id=tenant_id)
    return success_response(data=file_entity, message="Set file metadata success.")
