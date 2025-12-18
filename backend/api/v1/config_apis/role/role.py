### Role configuration API ###

import traceback
from typing import List
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.knowledgebase.user_role import (
    RoleEntity,
    PermissionEntity,
    UserRoleEntity,
)
from db.db_context import get_db_session
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException
from service.tool.role_service import RoleService
from service.injection import get_role_service
from loguru import logger
from service.injection import get_tenant_id

role_router = APIRouter()

# 角色API

@role_router.post("", response_model=ResponseModel[RoleEntity])
async def create_role(
    role: RoleEntity,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    try:
        role = await role_service.create_role(role=role, tenant_id=tenant_id)
        return success_response(data=role, message="添加角色成功。")
    except ValueError as e:
        logger.error(f"Failed to create role: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to create role: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"创建角色失败: '{e}'.")



@role_router.get("")
async def list_roles(
    name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    try:
        if name:
            # If name is provided, get single role by name
            role = await role_service.get_role_by_name(name)
            if not role:
                raise ApiException(
                    code=404, message=f"查询角色失败: '{name}'不存在。"
                )
            return success_response(data=role, message="查询角色成功。")
        else:
            # List all roles with pagination
            roles = await role_service.list_roles(page=page, size=size, tenant_id=tenant_id)
            return success_response(data=roles, message="查询角色列表成功")
    except Exception as e:
        logger.error(f"Failed to list roles: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"查询角色失败: '{e}'.")


@role_router.delete("/{role_id}")
async def delete_role(
    role_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    try:
        await role_service.delete_role(role_id=role_id, tenant_id=tenant_id)
        logger.info(f"角色 {role_id} 已删除.")
        return success_response(message=f"角色{role_id}删除成功。")
    except ValueError as e:
        logger.error(f"Failed to delete role: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete role: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"删除角色失败: '{e}'.")


# 用户-角色 API
@role_router.post("/user_roles", response_model=ResponseModel[UserRoleEntity])
async def create_user_role(
    user_role: UserRoleEntity,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    try:
        user_role = await role_service.create_user_role(user_role=user_role, tenant_id=tenant_id)
        return success_response(data=user_role, message="添加用户角色成功。")
    except ValueError as e:
        logger.error(f"Failed to create user role: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to create user role: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"创建用户角色失败: '{e}'.")



@role_router.get("/user_roles")
async def list_user_roles(
    user_id: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    try:
        user_roles = await role_service.list_user_roles(page=page, size=size, user_id=user_id, tenant_id=tenant_id)
        return success_response(data=user_roles, message="查询用户角色列表成功")
    except Exception as e:
        logger.error(f"Failed to list user roles: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"查询用户角色列表失败: '{e}'.")


@role_router.delete("/user_roles/{user_role_id}")
async def delete_user_role(
    user_role_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    try:
        await role_service.delete_user_role(user_role_id=user_role_id, tenant_id=tenant_id)
        logger.info(f"用户角色 {user_role_id} 已删除.")
        return success_response(message=f"用户角色{user_role_id}删除成功。")
    except ValueError as e:
        logger.error(f"Failed to delete user role: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete user role: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"删除用户角色失败: '{e}'.")



# Permission API
@role_router.post("/permissions", response_model=ResponseModel[PermissionEntity])
async def create_permission(
    permission: PermissionEntity,
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        permission = await role_service.create_permission(permission, tenant_id=tenant_id)
        await session.commit()
        await session.refresh(permission)
        return success_response(data=permission, message="添加权限成功。")
    except ValueError as e:
        logger.error(f"Failed to create permission: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to create permission: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=400, message=f"创建权限失败: '{e}'.")



@role_router.get("/permissions")
async def list_permissions(
    name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        permissions = await role_service.list_permissions(page=page, size=size, name=name, tenant_id=tenant_id)
        return success_response(data=permissions, message="查询权限成功")
    except Exception as e:
        logger.error(f"Failed to list permissions: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"查询权限失败: '{e}'.")



class UpdateRolePermission(BaseModel):
    role_ids: List[str]



@role_router.post("/permissions/files/{file_id}")
async def set_file_permission(
    file_id: str,
    update_request: UpdateRolePermission,
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        new_permissions = await role_service.set_file_permissions(
            file_id, update_request.role_ids, tenant_id=tenant_id
        )
        return success_response(data=new_permissions, message="更新权限成功")
    except ValueError as e:
        logger.error(f"设置文件权限失败: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as ex:
        logger.error(f"设置文件权限失败: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"设置文件权限失败: {ex}")



@role_router.delete("/permissions/{permission_id}")
async def delete_permission(
    permission_id: str,
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
    tenant_id: str = Depends(get_tenant_id),
):
    try:
        await role_service.delete_permission(permission_id, tenant_id=tenant_id)
        logger.info(f"权限 {permission_id} 已删除.")
        return success_response(message=f"权限 {permission_id} 删除成功。")
    except ValueError as e:
        logger.error(f"Failed to delete permission: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete permission: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"删除权限失败: '{e}'.")
