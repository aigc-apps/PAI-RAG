### Knowledgebase configuration API ###
import traceback
from datetime import datetime, timezone
from typing import List, Tuple, Union
from fastapi import Depends, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import func, delete
from sqlalchemy.orm.attributes import flag_modified
from db.models.knowledgebase.file import KbFileEntity, MetadataEntryData
from db.models.knowledgebase.knowledgebase import KbEntity
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from db.models.knowledgebase.metadata import FileMetadataEntity, KbMetadataEntity, MetadataValueType, KbMetadataEntityCreate
from api.response_model import ResponseModel, success_response, error_response
from api.v1.config_apis.knowledgebase import knowledgebase_router
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
async def add_kb_metadata(
    kb_id: str,
    metadata_create: KbMetadataEntityCreate,
    session: AsyncSession = Depends(get_session),
):
    kb_entity = (await session.exec(
        select(KbEntity)
        .where(KbEntity.id == kb_id)
    )).first()
    if kb_entity is None:
        return error_response(404, f"知识库{kb_id}不存在。")

    try:
        metadata_entity = KbMetadataEntity.model_validate(metadata_create, update={"kb_id": kb_id})

        session.add(metadata_entity)
        await session.commit()
        await session.refresh(metadata_entity)
        return success_response(data=metadata_entity, message="元数据创建成功。")

    except IntegrityError as e:
        # TODO: 这里有bug,logger.exception没有打印错误调用栈
        logger.exception(f"创建元数据失败。\nIntegrityError:{e}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return error_response(code=400, message="创建元数据失败: 元数据名称已存在。")
        else:
            return error_response(code=400, message=f"创建元数据失败: {e}.")
    except Exception as e:
        logger.exception(f"创建元数据失败。\nException:{e}")
        await session.rollback()
        return error_response(code=400, message=f"创建元数据失败: {e}.")


@knowledgebase_router.get("/{kb_id}/metadata", response_model=ResponseModel[List[dict]])
async def list_metadata(
    kb_id: str,
    offset: int = 0,
    limit: int = Query(default=20, lte=1000),
    session: AsyncSession = Depends(get_session),
):
    # 子查询：统计每个 metadata_id 被引用的文件数量
    file_count_subquery = (
        select(
            FileMetadataEntity.metadata_id,
            func.count(FileMetadataEntity.id).label('count')
        )
        .where(FileMetadataEntity.kb_id == kb_id)
        .group_by(FileMetadataEntity.metadata_id)
        .subquery()
    )

    # 主查询：LEFT JOIN 获取文件数量
    query = (
        select(
            KbMetadataEntity,
            func.coalesce(file_count_subquery.c.count, 0).label('count')
        )
        .outerjoin(
            file_count_subquery,
            KbMetadataEntity.id == file_count_subquery.c.metadata_id
        )
        .where(KbMetadataEntity.kb_id == kb_id)
        .offset(offset)
        .limit(limit)
    )

    results = await session.exec(query)
    metadata_with_counts = results.all()

    # 构建返回结果，将 count 添加到 metadata 字典中
    metadata_list = []
    for metadata_entity, count in metadata_with_counts:
        metadata_dict = metadata_entity.model_dump()
        metadata_dict['count'] = count
        metadata_list.append(metadata_dict)

    logger.info(f"Listing knowledgebase metadata infos: get {len(metadata_list)} in total.")

    return success_response(data=metadata_list, message="查询元数据成功。")


@knowledgebase_router.put("/{kb_id}/metadata/{metadata_id}", response_model=ResponseModel[List[KbMetadataEntity]])
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
        return error_response(code=400, message=f"更新元数据失败: 元数据'{metadata_id}'不存在。")

    old_name = metadata_entity.name
    old_value_type = metadata_entity.value_type
    new_name = new_metadata_entity.name
    new_value_type = new_metadata_entity.value_type

    if old_name == new_name and old_value_type == new_value_type:
        logger.info(f"没有数据更新，更新元数据成功: {metadata_entity}.")
        return success_response(data=metadata_entity, message="更新元数据成功。")

    try:
        # 更新 metadata_entity
        metadata_entity.name = new_name
        metadata_entity.value_type = new_value_type
        metadata_entity.description = new_metadata_entity.description
        metadata_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        session.add(metadata_entity)


        # 查找所有使用该 metadata 的文件
        file_metadata_entities = await session.exec(
            select(FileMetadataEntity)
            .where(FileMetadataEntity.metadata_id == metadata_id)
            .where(FileMetadataEntity.kb_id == kb_id)
        )
        file_metadata_list = file_metadata_entities.all()

        # 获取所有相关的文件
        file_ids = [fme.file_id for fme in file_metadata_list]
        if file_ids:
            files = await session.exec(
                select(KbFileEntity)
                .where(KbFileEntity.id.in_(file_ids))
                .where(KbFileEntity.kb_id == kb_id)
            )
            file_list = files.all()

            # 如果名称改变了，需要更新所有相关文件的 file_metadata 中的键名
            if old_name != new_name:
                logger.info(f"更新文件的 file_metadata 中的键名: {old_name} -> {new_name}.")
                # 更新每个文件的 file_metadata，将旧键名改为新键名
                for file_entity in file_list:
                    if old_name in file_entity.file_metadata:
                        # 保留值，但使用新键名
                        value = file_entity.file_metadata[old_name]
                        # 如果 value_type 也改变了，需要验证和转换值
                        if old_value_type != new_value_type:
                            is_valid, converted_value = _validate_metadata_value(value, new_value_type)
                            if not is_valid:
                                # 如果值无法转换，跳过该文件（或使用默认值）
                                logger.warning(f"文件 {file_entity.id} 的元数据值 '{value}' 无法转换为新类型 '{new_value_type}'，跳过更新")
                                continue
                            value = converted_value

                        # 删除旧键，添加新键
                        file_entity.file_metadata.pop(old_name, None)
                        file_entity.file_metadata[new_name] = value
                        # 标记 JSON 字段已修改，确保 SQLAlchemy 检测到变化
                        flag_modified(file_entity, "file_metadata")
                        session.add(file_entity)

            # 如果 value_type 改变了, 需要验证和转换所有相关文件的值
            if old_value_type != new_value_type:
                logger.info(f"更新文件的 file_metadata 中的值类型: {old_value_type} -> {new_value_type}.")
                # 更新每个文件的 file_metadata 中的值类型
                for file_entity in file_list:
                    if new_name in file_entity.file_metadata:
                        value = file_entity.file_metadata[new_name]
                        is_valid, converted_value = _validate_metadata_value(value, new_value_type)
                        if is_valid:
                            file_entity.file_metadata[new_name] = converted_value
                            # 标记 JSON 字段已修改，确保 SQLAlchemy 检测到变化
                            flag_modified(file_entity, "file_metadata")
                            session.add(file_entity)
                        else:
                            logger.warning(f"文件 {file_entity.id} 的元数据值 '{value}' 无法转换为新类型 '{new_value_type}'，跳过更新")

        await session.commit()
        await session.refresh(metadata_entity)
        logger.info(f"Update knowledgebase metadata infos successfully {metadata_entity}.")
        return success_response(data=metadata_entity, message=f"更新元数据成功，{len(file_list)}个相关文件的 file_metadata 已更新。")
    except Exception as e:
        logger.exception(f"更新元数据失败。\nException:{e}")
        await session.rollback()
        return error_response(code=400, message=f"更新元数据失败: {e}.")

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
        return error_response(code=400, message=f"删除元数据失败: 元数据'{metadata_id}'不存在。")

    try:
        metadata_name = metadata_entity.name

        # 第一步：查找所有使用该 metadata 的文件
        file_metadata_entities = await session.exec(
            select(FileMetadataEntity)
            .where(FileMetadataEntity.metadata_id == metadata_id)
            .where(FileMetadataEntity.kb_id == kb_id)
        )
        file_metadata_list = file_metadata_entities.all()

        # 第二步：从所有相关文件的 file_metadata JSON 字段中删除该 metadata
        if file_metadata_list:
            file_ids = [fme.file_id for fme in file_metadata_list]
            files = await session.exec(
                select(KbFileEntity)
                .where(KbFileEntity.id.in_(file_ids))
                .where(KbFileEntity.kb_id == kb_id)
            )
            file_list = files.all()

            # 从每个文件的 file_metadata 中删除该 metadata 的键值对
            for file_entity in file_list:
                if metadata_name in file_entity.file_metadata:
                    file_entity.file_metadata.pop(metadata_name)
                    file_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    # 标记 JSON 字段已修改，确保 SQLAlchemy 检测到变化
                    flag_modified(file_entity, "file_metadata")
                    session.add(file_entity)

        # 第三步：删除所有相关的 FileMetadataEntity 记录
        await session.execute(
            delete(FileMetadataEntity)
            .where(FileMetadataEntity.metadata_id == metadata_id)
            .where(FileMetadataEntity.kb_id == kb_id)
        )

        # 第四步：删除 KbMetadataEntity
        await session.delete(metadata_entity)
        await session.commit()

        logger.info(f"删除元数据成功，{len(file_list)}个相关文件的 file_metadata 已删除。")
        return success_response(data=metadata_entity, message="删除元数据成功，{len(file_list)}个相关文件的 file_metadata 已删除。")
    except Exception:
        logger.exception(f"删除元数据 {metadata_id} 失败: {traceback.format_exc()}.")
        await session.rollback()
        return error_response(code=500, message=f"删除元数据失败: {traceback.format_exc()}.")

def _validate_metadata_value(value: Union[str, int, float], value_type: str) -> Tuple[bool, Union[str, int, float]]:
    """
    验证元数据值是否匹配指定的类型
    返回: (是否有效, 转换后的值)
    """
    try:
        if value_type == MetadataValueType.STRING:
            return True, str(value)
        elif value_type == MetadataValueType.NUMBER:
            if isinstance(value, (int, float)):
                return True, float(value)
            try:
                return True, float(value)
            except (ValueError, TypeError):
                return False, value
        elif value_type == MetadataValueType.DATETIME:
            # datetime 类型也存储为数字（时间戳）或字符串
            if isinstance(value, (int, float)):
                return True, float(value)
            try:
                return True, float(value)
            except (ValueError, TypeError):
                return False, value  # 如果无法转换为数字，则作为字符串存储
        else:
            return False, value
    except Exception:
        return False, value


@knowledgebase_router.post("/{kb_id}/files/{file_id}/metadata", response_model=ResponseModel[KbFileEntity])
async def set_file_metadata(
    kb_id: str,
    file_id: str,
    entry_data: MetadataEntryData,
    session: AsyncSession = Depends(get_session)
):
    logger.info(f"Updating metadata {entry_data} for {file_id}.")
    try:
        # 第一步：验证文件是否存在
        file_entity = (await session.exec(
            select(KbFileEntity)
            .where(KbFileEntity.kb_id == kb_id)
            .where(KbFileEntity.id == file_id)
        )).first()
        if file_entity is None:
            return error_response(404, f"文件{file_id}不存在。")

        # 第二步：获取所有元数据配置，构建 name -> (id, value_type) 的映射
        metadata_results = await session.exec(
            select(KbMetadataEntity).where(KbMetadataEntity.kb_id == kb_id)
        )
        metadata_list = metadata_results.all()
        metadata_map = {metadata.name: (metadata.id, metadata.value_type) for metadata in metadata_list}

        # 第三步：验证 entry_data 中的 metadata name 是否存在于 KbMetadataEntity 中
        invalid_names = []
        for entry in entry_data.entries:
            if entry.name not in DEFAULT_METADATA_KEYS and entry.name not in metadata_map:
                invalid_names.append(entry.name)

        if invalid_names:
            return error_response(
                400,
                f"以下元数据名称不存在于知识库配置中: {', '.join(invalid_names)}"
            )

        # 第四步：验证数据类型和 value 是否合法
        validated_entries = []
        for entry in entry_data.entries:
            # 跳过默认的 metadata keys
            if entry.name in DEFAULT_METADATA_KEYS:
                continue

            metadata_id, value_type = metadata_map[entry.name]
            is_valid, converted_value = _validate_metadata_value(entry.value, value_type)

            if not is_valid:
                return error_response(
                    400,
                    f"元数据 '{entry.name}' 的值 '{entry.value}' 无法转换为类型 '{value_type}'"
                )

            validated_entries.append({
                'name': entry.name,
                'value': converted_value,
                'metadata_id': metadata_id
            })

        # 第五步：直接用 entry_data 覆盖 file_metadata（保留 default_metadata_keys）
        # 保留原有的 default_metadata_keys
        new_metadata = {k: v for k, v in file_entity.file_metadata.items() if k in DEFAULT_METADATA_KEYS}
        # 添加验证后的元数据
        for entry in validated_entries:
            new_metadata[entry['name']] = entry['value']

        # 第六步：更新相关的 FileMetadataEntity
        # 先删除所有现有的 FileMetadataEntity
        await session.execute(
            delete(FileMetadataEntity)
            .where(FileMetadataEntity.kb_id == kb_id)
            .where(FileMetadataEntity.file_id == file_id)
        )

        # 然后添加新的 FileMetadataEntity
        for entry in validated_entries:
            session.add(
                FileMetadataEntity(
                    file_id=file_entity.id,
                    metadata_id=entry['metadata_id'],
                    kb_id=kb_id,
                )
            )

        # 更新文件的 file_metadata
        file_entity.file_metadata = new_metadata
        session.add(file_entity)
        await session.commit()
        await session.refresh(file_entity)
        logger.info(f"File meta {file_entity.id} updated to {file_entity.file_metadata}.")

        return success_response(data=file_entity, message="文件元数据更新成功。")

    except IntegrityError as e:
        logger.exception(f"更新文件元数据失败。\nIntegrityError:{e}")
        await session.rollback()
        return error_response(code=400, message=f"更新文件元数据失败: {e}.")
    except Exception as e:
        logger.exception(f"更新文件元数据失败。\nException:{e}")
        await session.rollback()
        return error_response(code=400, message=f"更新文件元数据失败: {e}.")
