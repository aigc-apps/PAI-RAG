"""Role Service layer for database operations."""

from typing import Optional, List
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from db.models.knowledgebase.user_role import (
    RoleEntity,
    PermissionEntity,
    UserRoleEntity,
)
from common.chat.response_model import PagedResult


class RoleService:
    """Service layer for Role, UserRole, and Permission entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize RoleService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    # ========== Role Operations ==========

    async def get_role(self, role_id: str, tenant_id: str) -> Optional[RoleEntity]:
        """
        Get a single Role entity by ID.

        Args:
            role_id: Role entity ID

        Returns:
            RoleEntity if found, None otherwise
        """
        result = await self.session.exec(select(RoleEntity).where(RoleEntity.id == role_id, RoleEntity.tenant_id == tenant_id))
        return result.first()

    async def get_role_by_name(self, name: str, tenant_id: str) -> Optional[RoleEntity]:
        """
        Get a single Role entity by name.

        Args:
            name: Role name

        Returns:
            RoleEntity if found, None otherwise
        """
        statement = select(RoleEntity).where(RoleEntity.name == name, RoleEntity.tenant_id == tenant_id)
        result = await self.session.exec(statement)
        return result.first()

    async def list_roles(
        self,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
        name: Optional[str] = None,
    ) -> PagedResult[List[RoleEntity]]:
        """
        List Role entities with pagination and optional filtering.

        Args:
            page: Page number (1-indexed)
            size: Page size
            name: Optional filter for role name

        Returns:
            PagedResult containing list of RoleEntity and pagination metadata
        """
        # Build base query
        base_query = select(RoleEntity).where(RoleEntity.tenant_id == tenant_id)

        # Add name filter if provided
        if name is not None:
            base_query = base_query.where(RoleEntity.name == name)

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = base_query.offset(offset).limit(size)
        results = await self.session.exec(paginated_query)
        roles = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=roles,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_role(self, role: RoleEntity, tenant_id: str) -> RoleEntity:
        """
        Create a new Role entity.
        Note: Caller is responsible for committing the session.

        Args:
            role: Role entity to create

        Returns:
            Created RoleEntity (not yet committed)

        Raises:
            ValueError: If name already exists (IntegrityError converted)
        """
        role.tenant_id = tenant_id
        self.session.add(role)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(role)

            logger.info(f"Created Role entity: {role.id} (name: {role.name})")
            return role

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Role: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(f"Role name '{role.name}' already exists.") from e
            else:
                raise ValueError(f"Role creation failed: {e}") from e

    async def delete_role(self, role_id: str, tenant_id: str) -> None:
        """
        Delete a Role entity.
        Note: Caller is responsible for committing the session.

        Args:
            role_id: Role entity ID

        Raises:
            ValueError: If Role entity not found
        """
        result = await self.session.exec(select(RoleEntity).where(RoleEntity.id == role_id, RoleEntity.tenant_id == tenant_id))
        role = result.first()
        if not role:
            raise ValueError(f"Role '{role_id}' does not exist.")

        # Delete from database (staged, not committed)
        await self.session.delete(role)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted Role entity: {role_id} (name: {role.name})")

    # ========== UserRole Operations ==========

    async def get_user_role(self, user_role_id: str, tenant_id: str) -> Optional[UserRoleEntity]:
        """
        Get a single UserRole entity by ID.

        Args:
            user_role_id: UserRole entity ID

        Returns:
            UserRoleEntity if found, None otherwise
        """
        result = await self.session.exec(select(UserRoleEntity).where(UserRoleEntity.id == user_role_id, UserRoleEntity.tenant_id == tenant_id))
        return result.first()

    async def list_user_roles(
        self,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
        user_id: Optional[str] = None,
    ) -> PagedResult[List[UserRoleEntity]]:
        """
        List UserRole entities with pagination and optional filtering.

        Args:
            page: Page number (1-indexed)
            size: Page size
            user_id: Optional filter for user_id

        Returns:
            PagedResult containing list of UserRoleEntity and pagination metadata
        """
        # Build base query
        base_query = select(UserRoleEntity).where(UserRoleEntity.tenant_id == tenant_id)

        # Add user_id filter if provided
        if user_id is not None:
            base_query = base_query.where(UserRoleEntity.user_id == user_id)

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = base_query.offset(offset).limit(size)
        results = await self.session.exec(paginated_query)
        user_roles = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=user_roles,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_user_role(self, user_role: UserRoleEntity, tenant_id: str) -> UserRoleEntity:
        """
        Create a new UserRole entity.
        Note: Caller is responsible for committing the session.

        Args:
            user_role: UserRole entity to create

        Returns:
            Created UserRoleEntity (not yet committed)

        Raises:
            ValueError: If user_id-role_id combination already exists (IntegrityError converted)
        """
        user_role.tenant_id = tenant_id
        self.session.add(user_role)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(user_role)

            logger.info(
                f"Created UserRole entity: {user_role.id} (user_id: {user_role.user_id}, role_id: {user_role.role_id})"
            )
            return user_role

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating UserRole: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"User role '{user_role.user_id} - {user_role.role_id}' already exists."
                ) from e
            else:
                raise ValueError(f"User role creation failed: {e}") from e

    async def delete_user_role(self, user_role_id: str, tenant_id: str) -> None:
        """
        Delete a UserRole entity.
        Note: Caller is responsible for committing the session.

        Args:
            user_role_id: UserRole entity ID

        Raises:
            ValueError: If UserRole entity not found
        """
        result = await self.session.exec(select(UserRoleEntity).where(UserRoleEntity.id == user_role_id, UserRoleEntity.tenant_id == tenant_id))
        user_role = result.first()
        if not user_role:
            raise ValueError(f"User role '{user_role_id}' does not exist.")

        # Delete from database (staged, not committed)
        await self.session.delete(user_role)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(
            f"Deleted UserRole entity: {user_role_id} (user_id: {user_role.user_id}, role_id: {user_role.role_id})"
        )

    # ========== Permission Operations ==========

    async def get_permission(
        self, permission_id: str, tenant_id: str
    ) -> Optional[PermissionEntity]:
        """
        Get a single Permission entity by ID.

        Args:
            permission_id: Permission entity ID

        Returns:
            PermissionEntity if found, None otherwise
        """
        result = await self.session.exec(select(PermissionEntity).where(PermissionEntity.id == permission_id, PermissionEntity.tenant_id == tenant_id))
        return result.first()

    async def list_permissions(
        self,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
        name: Optional[str] = None,
    ) -> PagedResult[List[PermissionEntity]]:
        """
        List Permission entities with pagination and optional filtering.

        Args:
            page: Page number (1-indexed)
            size: Page size
            name: Optional filter for permission name

        Returns:
            PagedResult containing list of PermissionEntity and pagination metadata
        """
        # Build base query
        base_query = select(PermissionEntity).where(PermissionEntity.tenant_id == tenant_id)

        # Add name filter if provided
        if name is not None:
            base_query = base_query.where(PermissionEntity.name == name)

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = base_query.offset(offset).limit(size)
        results = await self.session.exec(paginated_query)
        permissions = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=permissions,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_permission(
        self, permission: PermissionEntity, tenant_id: str
    ) -> PermissionEntity:
        """
        Create a new Permission entity.
        Note: Caller is responsible for committing the session.

        Args:
            permission: Permission entity to create

        Returns:
            Created PermissionEntity (not yet committed)

        Raises:
            ValueError: If name-role_id combination already exists (IntegrityError converted)
        """
        permission.tenant_id = tenant_id
        self.session.add(permission)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(permission)

            logger.info(
                f"Created Permission entity: {permission.id} (name: {permission.name}, role_id: {permission.role_id})"
            )
            return permission

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Permission: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"Permission name '{permission.name}' with role '{permission.role_id}' combination already exists."
                ) from e
            else:
                raise ValueError(f"Permission creation failed: {e}") from e

    async def delete_permission(self, permission_id: str, tenant_id: str) -> None:
        """
        Delete a Permission entity.
        Note: Caller is responsible for committing the session.

        Args:
            permission_id: Permission entity ID

        Raises:
            ValueError: If Permission entity not found
        """
        result = await self.session.exec(select(PermissionEntity).where(PermissionEntity.id == permission_id, PermissionEntity.tenant_id == tenant_id))
        permission = result.first()
        if not permission:
            raise ValueError(f"Permission '{permission_id}' does not exist.")

        # Delete from database (staged, not committed)
        await self.session.delete(permission)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(
            f"Deleted Permission entity: {permission_id} (name: {permission.name}, role_id: {permission.role_id})"
        )

    async def set_file_permissions(
        self, file_id: str, role_ids: List[str], tenant_id: str
    ) -> List[PermissionEntity]:
        """
        Set permissions for a file by updating role associations.
        This will:
        1. Delete existing permissions for the file that are not in the new role_ids
        2. Create new permissions for role_ids that don't already exist

        Note: Caller is responsible for committing the session.

        Args:
            file_id: File ID to set permissions for
            role_ids: List of role IDs that should have permission for this file

        Returns:
            List of newly created PermissionEntity (not yet committed)
        """
        # Get existing permissions for this file
        existing_permissions_result = await self.session.exec(
            select(PermissionEntity).where(PermissionEntity.name == file_id, PermissionEntity.tenant_id == tenant_id)
        )
        existing_permissions = list(existing_permissions_result.all())

        new_role_ids = set(role_ids)
        old_role_ids = {p.role_id for p in existing_permissions}

        # Create new permissions for roles that don't have permission yet
        new_permissions = []
        for role_id in role_ids:
            if role_id not in old_role_ids:
                permission = PermissionEntity(
                    name=file_id,
                    description=f"Read permission for file {file_id}",
                    role_id=role_id,
                    tenant_id=tenant_id,
                )
                self.session.add(permission)
                new_permissions.append(permission)

        # Delete permissions for roles that are no longer in the list
        for old_permission in existing_permissions:
            if old_permission.role_id not in new_role_ids:
                await self.session.delete(old_permission)

        try:
            # Flush to get IDs for new permissions, but don't commit
            await self.session.flush()
            for permission in new_permissions:
                await self.session.refresh(permission)

            logger.info(
                f"Set file permissions for file {file_id}: {len(new_permissions)} new permissions created, {len(existing_permissions) - len([p for p in existing_permissions if p.role_id in new_role_ids])} permissions deleted"
            )
            return new_permissions

        except IntegrityError as e:
            logger.error(f"IntegrityError when setting file permissions: {e.orig}")
            raise ValueError(f"Failed to set file permissions: {e}") from e
