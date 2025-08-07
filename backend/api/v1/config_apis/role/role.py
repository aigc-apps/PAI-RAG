### Embedding configuration API ###

import traceback
from typing import List
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.knowledgebase.user_role import (
    RoleEntity,
    PermissionEntity,
    UserRoleEntity,
)
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from api.response_model import PagedResult, ResponseModel, success_response, error_response
from api.v1.utils.paginate import get_pagination_meta
from loguru import logger

role_router = APIRouter()

# 角色API

@role_router.post("", response_model=ResponseModel[RoleEntity])
async def create_role(
    role: RoleEntity, session: AsyncSession = Depends(get_session)
):
    try:
        session.add(role)
        await session.commit()
        await session.refresh(role)
        return success_response(data=role, message="添加角色成功。")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add role: {e.orig}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return error_response(
                    code=400, message=f"创建角色失败: 角色'{role.name}'已存在."
                )
        else:
            return error_response(code=400, message=f"创建角色失败: '{e}'.")
    except Exception as e:
        await session.rollback()
        return error_response(code=400, message=f"创建角色失败: '{e}'.")



@role_router.get("")
async def list_roles(
    name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    if not name:
        total_results = await session.exec(
            select(func.count()).select_from(
                select(RoleEntity)
            )
        )
        total_num = total_results.one_or_none()
        pagination = get_pagination_meta(page, size, total_num)
        sql_results = await session.exec(select(RoleEntity).offset(pagination.offset).limit(size))
        role_entities = sql_results.all()

        return success_response(
            data=PagedResult(
                items=role_entities,
                total=pagination.total,
                pages=pagination.pages,
                page=pagination.page,
                size=pagination.size,
            ),message="查询角色列表成功")
    else:
        statement = select(RoleEntity).where(
            RoleEntity.name == RoleEntity
        )
        role = (await session.exec(statement)).first()
        if not role:
            return error_response(
                code=404, message=f"查询角色失败: '{name}'不存在。"
            )
        return success_response(data=role, message="查询角色成功。")


@role_router.delete("/{role_id}")
async def delete_role(
    role_id: str,
    session: AsyncSession = Depends(get_session),
):
    role = await session.get(RoleEntity, role_id)
    if not role:
        return error_response(
                code=404, message=f"删除role失败: 角色'{role_id}'不存在。"
            )

    await session.delete(role)
    await session.commit()

    logger.info(f"角色 {role_id} 已删除.")
    return success_response(message=f"角色{role_id}删除成功。")


# 用户-角色 API
@role_router.post("/user_roles", response_model=ResponseModel[UserRoleEntity])
async def create_user_role(
    user_role: UserRoleEntity, session: AsyncSession = Depends(get_session)
):
    try:
        session.add(user_role)
        await session.commit()
        await session.refresh(user_role)
        return success_response(data=user_role, message="添加用户角色成功。")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add user role: {e.orig}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return error_response(
                    code=400, message=f"创建用户角色失败: 用户角色'{user_role.user_id} - {user_role.role_id}'已存在."
                )
        else:
            return error_response(code=400, message=f"创建用户角色失败: '{e}'.")
    except Exception as e:
        await session.rollback()
        return error_response(code=400, message=f"创建用户角色失败: '{e}'.")



@role_router.get("/user_roles")
async def list_user_roles(
    user_id: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    if not user_id:
        total_results = await session.exec(
            select(func.count()).select_from(
                select(UserRoleEntity)
            )
        )
        total_num = total_results.one_or_none()
        pagination = get_pagination_meta(page, size, total_num)
        sql_results = await session.exec(select(UserRoleEntity).offset(pagination.offset).limit(size))
        role_entities = sql_results.all()
    else:
        total_results = await session.exec(
            select(func.count()).select_from(
                select(UserRoleEntity).where(UserRoleEntity.user_id == user_id)
            )
        )
        total_num = total_results.one_or_none()
        pagination = get_pagination_meta(page, size, total_num)
        sql_results = await session.exec(select(UserRoleEntity).where(UserRoleEntity.user_id == user_id).offset(pagination.offset).limit(size))
        role_entities = sql_results.all()

    return success_response(
        data=PagedResult(
            items=role_entities,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),message="查询用户角色列表成功")


@role_router.delete("/user_roles/{user_role_id}")
async def delete_user_role(
    user_role_id: str,
    session: AsyncSession = Depends(get_session),
):
    user_role = await session.get(UserRoleEntity, user_role_id)
    if not user_role:
        return error_response(
                code=404, message=f"删除用户角色失败: 角色'{user_role_id}'不存在。"
            )

    await session.delete(user_role)
    await session.commit()

    logger.info(f"用户角色 {user_role_id} 已删除.")
    return success_response(message=f"用户角色{user_role_id}删除成功。")



# Permission API
@role_router.post("/permissions", response_model=ResponseModel[PermissionEntity])
async def create_permission(
    permission: PermissionEntity, session: AsyncSession = Depends(get_session)
):
    try:
        session.add(permission)
        await session.commit()
        await session.refresh(permission)
        return success_response(data=permission, message="添加权限成功。")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add permission: {e.orig}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return error_response(
                    code=400, message=f"创建权限失败: 权限名称'{permission.name}'已存在."
                )
        else:
            return error_response(code=400, message=f"创建权限失败: '{e}'.")
    except Exception as e:
        await session.rollback()
        return error_response(code=400, message=f"创建权限失败: '{e}'.")



@role_router.get("/permissions")
async def list_permissions(
    name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    if not name:
        total_results = await session.exec(
            select(func.count()).select_from(
                select(PermissionEntity)
            )
        )
        total_num = total_results.one_or_none()
        pagination = get_pagination_meta(page, size, total_num)
        sql_results = await session.exec(select(PermissionEntity).offset(pagination.offset).limit(size))
        role_entities = sql_results.all()
    else:
        total_results = await session.exec(
            select(func.count()).select_from(
                select(PermissionEntity).where(PermissionEntity.name == name)
            )
        )
        total_num = total_results.one_or_none()
        pagination = get_pagination_meta(page, size, total_num)
        sql_results = await session.exec(select(PermissionEntity).where(PermissionEntity.name == name).offset(pagination.offset).limit(size))
        role_entities = sql_results.all()

    return success_response(
        data=PagedResult(
            items=role_entities,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),message="查询权限成功")



class UpdateRolePermission(BaseModel):
    role_ids: List[str]



@role_router.post("/permissions/files/{file_id}")
async def set_file_permission(
    file_id: str,
    update_request: UpdateRolePermission,
    session: AsyncSession = Depends(get_session),
):
    try:
        exitsing_permissions = (await session.exec(
            select(PermissionEntity).where(PermissionEntity.name == file_id)
        )).all()
        new_role_ids =set(update_request.role_ids)
        old_role_ids = set([p.role_id for p in exitsing_permissions])

        new_permissions = [
            PermissionEntity(
                name=file_id,
                description=f"Read permission for file {file_id}",
                role_id=role_id,
            )
            for role_id in update_request.role_ids if role_id not in old_role_ids
        ]

        for old_permission in exitsing_permissions:
            if old_permission.role_id not in new_role_ids:
                await session.delete(old_permission)
        for permission in new_permissions:
            session.add(permission)
        await session.commit()

        return success_response(data=new_permissions, message="更新权限成功")
    except Exception as ex:
        logger.error(f"设置文件权限失败: {traceback.format_exc()}")
        return error_response(message=f"设置文件权限失败: {ex}", code=400)



@role_router.delete("/permissions/{permission_id}")
async def delete_permission(
    permission_id: str,
    session: AsyncSession = Depends(get_session),
):
    user_role = await session.get(PermissionEntity, permission_id)
    if not user_role:
        return error_response(
                code=404, message=f"删除权限失败: 权限'{permission_id}'不存在。"
            )

    await session.delete(user_role)
    await session.commit()

    logger.info(f"权限 {permission_id} 已删除.")
    return success_response(message=f"权限 {permission_id} 删除成功。")
