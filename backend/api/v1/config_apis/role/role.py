### Role configuration API ###

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
from api.api_exception import ApiException, handle_api_exceptions
from service.tool.role_service import RoleService
from service.injection import get_role_service
from service.injection import get_tenant_id

role_router = APIRouter()

# 角色API

@role_router.post("", response_model=ResponseModel[RoleEntity])
@handle_api_exceptions(action="create role", default_code=400)
async def create_role(
    role: RoleEntity,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    role = await role_service.create_role(role=role, tenant_id=tenant_id)
    return success_response(data=role, message="Create role success.")



@role_router.get("")
@handle_api_exceptions(action="list roles", default_code=400)
async def list_roles(
    name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    if name:
        # If name is provided, get single role by name
        role = await role_service.get_role_by_name(name)
        if not role:
            raise ApiException(
                code=404, message=f"Get role failed: '{name}' does not exist."
            )
        return success_response(data=role, message="Get role successful.")
    # List all roles with pagination
    roles = await role_service.list_roles(page=page, size=size, tenant_id=tenant_id)
    return success_response(data=roles, message="List role successful")


@role_router.delete("/{role_id}")
@handle_api_exceptions(action="delete role", default_code=400)
async def delete_role(
    role_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    await role_service.delete_role(role_id=role_id, tenant_id=tenant_id)
    return success_response(message=f"Role {role_id} deletion successful.")


# 用户-角色 API
@role_router.post("/user_roles", response_model=ResponseModel[UserRoleEntity])
@handle_api_exceptions(action="create user role", default_code=400)
async def create_user_role(
    user_role: UserRoleEntity,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    user_role = await role_service.create_user_role(user_role=user_role, tenant_id=tenant_id)
    return success_response(data=user_role, message="Create user role successful.")



@role_router.get("/user_roles")
@handle_api_exceptions(action="list user roles", default_code=400)
async def list_user_roles(
    user_id: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    user_roles = await role_service.list_user_roles(page=page, size=size, user_id=user_id, tenant_id=tenant_id)
    return success_response(data=user_roles, message="List user roles successful")


@role_router.delete("/user_roles/{user_role_id}")
@handle_api_exceptions(action="delete user role", default_code=400)
async def delete_user_role(
    user_role_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
):
    await role_service.delete_user_role(user_role_id=user_role_id, tenant_id=tenant_id)
    return success_response(message=f"User role {user_role_id} deletion successful.")



# Permission API
@role_router.post("/permissions", response_model=ResponseModel[PermissionEntity])
@handle_api_exceptions(action="create permission", default_code=400)
async def create_permission(
    permission: PermissionEntity,
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
    tenant_id: str = Depends(get_tenant_id),
):
    permission = await role_service.create_permission(permission, tenant_id=tenant_id)
    await session.refresh(permission)
    return success_response(data=permission, message="Create permission successful.")



@role_router.get("/permissions")
@handle_api_exceptions(action="list permissions", default_code=400)
async def list_permissions(
    name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
    tenant_id: str = Depends(get_tenant_id),
):
    permissions = await role_service.list_permissions(page=page, size=size, name=name, tenant_id=tenant_id)
    return success_response(data=permissions, message="List permissions successful")



class UpdateRolePermission(BaseModel):
    role_ids: List[str]



@role_router.post("/permissions/files/{file_id}")
@handle_api_exceptions(action="set file permissions", default_code=400)
async def set_file_permission(
    file_id: str,
    update_request: UpdateRolePermission,
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
    tenant_id: str = Depends(get_tenant_id),
):
    new_permissions = await role_service.set_file_permissions(
        file_id, update_request.role_ids, tenant_id=tenant_id
    )
    return success_response(data=new_permissions, message="Update permissions successful")



@role_router.delete("/permissions/{permission_id}")
@handle_api_exceptions(action="delete permission", default_code=400)
async def delete_permission(
    permission_id: str,
    session: AsyncSession = Depends(get_db_session),
    role_service: RoleService = Depends(get_role_service),
    tenant_id: str = Depends(get_tenant_id),
):
    await role_service.delete_permission(permission_id, tenant_id=tenant_id)
    return success_response(message=f"Permission {permission_id} deletion successful.")
