from datetime import datetime, timezone
import uuid
from pydantic import model_validator, field_serializer
from sqlalchemy import Column, DateTime, UniqueConstraint, Text
from sqlmodel import Field, SQLModel
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional

class PermissionEntity(SQLModel, table=True):
    __tablename__ = "pai_permissions"
    __table_args__ = (UniqueConstraint("name", "role_id", "tenant_id", name="unique_role_permission"),)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    name: str = Field(default=None)
    role_id: str = Field(default=None, foreign_key="pai_roles.id", ondelete="CASCADE", max_length=64)
    description: str | None = Field(default=None, sa_column=Column(Text))

    @model_validator(mode='after')
    def set_defaults(self):
        if not self.id:
            self.id = uuid.uuid4().hex
        return self


class RoleEntity(SQLModel, table=True):
    __tablename__ = "pai_roles"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True, max_length=64)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    name: str = Field(default=None, max_length=255)
    description: str | None = Field(default=None, sa_column=Column(Text))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    @model_validator(mode='after')
    def set_defaults(self):
        if not self.id:
            self.id = uuid.uuid4().hex
        return self


class UserRoleEntity(SQLModel, table=True):
    __tablename__ = "pai_user_roles"
    __table_args__ = (UniqueConstraint("user_id", "role_id", "tenant_id", name="unique_user_role"),)


    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True, max_length=64)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    user_id: str = Field(default=None, max_length=64)
    role_id: str = Field(default=None, foreign_key="pai_roles.id", ondelete="CASCADE", max_length=64)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    @model_validator(mode='after')
    def set_defaults(self):
        if not self.id:
            self.id = uuid.uuid4().hex
        return self

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()
