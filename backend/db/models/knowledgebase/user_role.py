from datetime import datetime, timezone
import uuid
from pydantic import model_validator
from sqlalchemy import Column, DateTime, UniqueConstraint
from sqlmodel import Field, SQLModel


class PermissionEntity(SQLModel, table=True):
    __tablename__ = "pai_permissions"
    __table_args__ = (UniqueConstraint("name", "role_id", name="unique_role_permission"),)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    name: str = Field(default=None)
    role_id: str = Field(default=None, foreign_key="pai_roles.id", ondelete="CASCADE")
    description: str | None = Field(default=None)

    @model_validator(mode='after')
    def set_defaults(self):
        if not self.id:
            self.id = uuid.uuid4().hex
        return self


class RoleEntity(SQLModel, table=True):
    __tablename__ = "pai_roles"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    name: str = Field(default=None, unique=True)
    description: str | None = Field(default=None)

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
    __table_args__ = (UniqueConstraint("user_id", "role_id", name="unique_user_role"),)


    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True)
    user_id: str = Field(default=None)
    role_id: str = Field(default=None, foreign_key="pai_roles.id", ondelete="CASCADE")
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
