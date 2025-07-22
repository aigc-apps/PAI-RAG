### Knowledgebase configuration API ###
import asyncio
import traceback
from typing import List
from fastapi import Depends, Query
from fastapi.responses import JSONResponse
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models.knowledgebase.file import KbFileEntity, MetadataEntryData
from pairag.db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from pairag.db.models.knowledgebase.metadata import FileMetadataEntity, KbMetadataEntity, MetadataValueType
from pairag.mcp.providers.knowledgebase_provider import knowledgebase_provider
from pairag.api.response_model import ResponseModel, success_response, error_response
from pairag.api.agent.config.knowledgebase import knowledgebase_router
from loguru import logger


DEFAULT_METADATA_KEYS = [
    "file_name",
    "file_path",
    "file_size",
    "file_extension",
    "file_url",
    "doc_id",
  ]


@knowledgebase_router.post("/{kb_id}/metadata", response_model=ResponseModel[KbMetadataEntity])
async def set_kb_metadata(
    kb_id: str,
    metadata_entity: KbMetadataEntity,
    session: AsyncSession = Depends(get_session),
):
    try:
        assert metadata_entity.kb_id == kb_id, "metadata body中的kb_id不一致"
        session.add(metadata_entity)
        await session.commit()
        await session.refresh(metadata_entity)
        asyncio.create_task(knowledgebase_provider.refresh())
        return success_response(data=metadata_entity, message="元数据创建成功。")

    except IntegrityError as e:
        # TODO: 这里有bug,logger.exception没有打印错误调用栈
        logger.exception(f"创建元数据失败。\nIntegrityError:{e}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return JSONResponse(
                content=error_response(code=400, message="创建元数据失败: 元数据名称已存在。"),
                status_code=400,
            )
        else:
            return JSONResponse(
                content=error_response(code=400, message=f"创建元数据失败: {e}."),
                status_code=400,
            )
    except Exception as e:
        logger.exception(f"创建元数据失败。\nException:{e}")
        await session.rollback()
        return JSONResponse(
            content=error_response(code=400, message=f"创建元数据失败: {e}."),
            status_code=400,
        )

@knowledgebase_router.get("/{kb_id}/metadata", response_model=ResponseModel[List[KbMetadataEntity]])
async def list_metadata(
    kb_id: str,
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
    session: AsyncSession = Depends(get_session),
):
    metadata_results = await session.exec(
        select(KbMetadataEntity).where(KbMetadataEntity.kb_id == kb_id).offset(offset).limit(limit)
    )
    metadata_list = metadata_results.all()
    logger.info(f"Listing knowledgebase metadata infos: get {len(metadata_list)} in total.")

    return success_response(data=metadata_list, message="查询元数据成功。")


@knowledgebase_router.patch("/{kb_id}/metadata/{metadata_id}", response_model=ResponseModel[List[KbMetadataEntity]])
async def update_metadata(
    kb_id: str,
    metadata_id: str,
    new_metadata_entity: KbMetadataEntity,
    session: AsyncSession = Depends(get_session),
):
    metadata_entity = (await session.exec(
        select(KbMetadataEntity)
        .where(KbMetadataEntity.id == metadata_id)
        .where(KbMetadataEntity.kb_id == kb_id)
    )).first()
    if metadata_entity is None:
        return JSONResponse(
            content=error_response(code=400, message=f"更新元数据失败: 元数据'{metadata_id}'不存在。"),
            status_code=400,
        )


    metadata_entity.name = new_metadata_entity.name
    metadata_entity.value_type = new_metadata_entity.value_type
    metadata_entity.description = new_metadata_entity.description
    session.add(metadata_entity)
    # update all file metadata
    await session.refresh(metadata_entity)
    await session.commit()

    logger.info(f"Update knowledgebase metadata infos successfully {metadata_entity}.")

    return success_response(data=metadata_entity, message="更新元数据成功。")


@knowledgebase_router.delete("/{kb_id}/metadata/{metadata_id}", response_model=ResponseModel[KbMetadataEntity])
async def delete_metadata(
    kb_id: str,
    metadata_id: str,
    session: AsyncSession = Depends(get_session),
):
    metadata_entity = (await session.exec(
        select(KbMetadataEntity)
        .where(KbMetadataEntity.id == metadata_id)
        .where(KbMetadataEntity.kb_id == kb_id)
    )).first()
    if metadata_entity is None:
        return JSONResponse(
            content=error_response(code=400, message=f"删除元数据失败: 元数据'{metadata_entity.id}'不存在。"),
            status_code=400,
        )

    try:
        # TODO: 删除对应文件中的元数据
        await session.delete(metadata_entity)
        await session.commit()
        logger.info(f"删除元数据成功: {metadata_entity.id}.")
        return success_response(data=metadata_entity, message="删除元数据成功.")
    except Exception:
        logger.exception(f"删除元数据 {metadata_id} 失败: {traceback.format_exc()}.")
        return JSONResponse(
            content=error_response(code=500, message=f"删除元数据失败: {traceback.format_exc()}."),
            status_code=500,
        )


@knowledgebase_router.post("/{kb_id}/files/{file_id}/metadata", response_model=ResponseModel[KbFileEntity])
async def set_file_metadata(
    kb_id: str,
    file_id: str,
    entry_data: MetadataEntryData,
    session: AsyncSession = Depends(get_session)
):
    logger.info(f"Updating metadata {entry_data} for {file_id}.")
    try:
        metadata_results = await session.exec(
            select(KbMetadataEntity).where(KbMetadataEntity.kb_id == kb_id)
        )
        value_type_map = { metadata.name: metadata.value_type for metadata in metadata_results.all() }

        file_entity = (await session.exec(
            select(KbFileEntity)
            .where(KbFileEntity.kb_id == kb_id)
            .where(KbFileEntity.id == file_id)
        )).first()
        if file_entity is None:
            return error_response(404, f"文件{file_id}不存在。")

        valid_entries = [
           entry for entry in entry_data.entries if entry.name not in DEFAULT_METADATA_KEYS and entry.name in value_type_map
        ]

        file_metadata_entities = (
            await session.exec(
                select(FileMetadataEntity)
                .where(FileMetadataEntity.kb_id == kb_id)
                .where(FileMetadataEntity.file_id == file_id)
            )).all()

        eixsting_metadata_ids = set([
            entity.metadata_id for entity in file_metadata_entities
        ])

        new_metadata_ids = set()
        new_metadata = {k: v for k, v in file_entity.file_metadata.items() if k in DEFAULT_METADATA_KEYS}
        for entry in valid_entries:
            value_type = value_type_map[entry.name]
            if value_type == MetadataValueType.NUMBER or value_type == MetadataValueType.DATETIME:
                new_metadata[entry.name] = float(entry.value)
            else:
                new_metadata[entry.name] = entry.value

            new_metadata_ids.add(entry.metadata_id)
            # 新增的metadata key，需要更新到文件-metadata表
            if entry.metadata_id not in eixsting_metadata_ids:
                session.add(
                    FileMetadataEntity(
                        file_id=file_entity.id,
                        metadata_id=entry.metadata_id,
                        kb_id=kb_id,
                    )
                )

        ids_to_delete = new_metadata_ids - eixsting_metadata_ids
        for entity in file_metadata_entities:
            if entity.id in ids_to_delete:
                await session.delete(entity)

        print(new_metadata, file_entity.file_metadata)
        file_entity.file_metadata = new_metadata

        session.add(file_entity)
        await session.commit()
        await session.refresh(file_entity)
        logger.info(f"File meta {file_entity.id} updated to {file_entity.file_metadata}.")

        return success_response(data=file_entity, message="文件元数据更新成功。")

    except IntegrityError as e:
        # TODO: 这里有bug,logger.exception没有打印错误调用栈
        logger.exception(f"创建元数据失败。\nIntegrityError:{e}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return JSONResponse(
                content=error_response(code=400, message="创建元数据失败: 元数据名称已存在。"),
                status_code=400,
            )
        else:
            return JSONResponse(
                content=error_response(code=400, message=f"创建元数据失败: {e}."),
                status_code=400,
            )
    except Exception as e:
        logger.exception(f"创建元数据失败。\nException:{e}")
        await session.rollback()
        return JSONResponse(
            content=error_response(code=400, message=f"创建元数据失败: {e}."),
            status_code=400,
        )
