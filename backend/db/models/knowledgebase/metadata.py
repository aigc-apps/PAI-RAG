from datetime import datetime, timezone
import re
import uuid
from pydantic import field_validator, field_serializer
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, Enum, UniqueConstraint, Text
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional


class MetadataValueType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    DATETIME = "datetime"


class KbMetadataEntityCreate(SQLModel):
    name: str = Field(default=None, min_length=3, max_length=50)
    value_type: str = Field(default=MetadataValueType.STRING)
    description: str = Field(default='', sa_column=Column(Text))

    @field_validator("name")
    def validate_metadata_name_format(cls, v):
        # 使用正则表达式检查是否只包含字母、数字、下划线和短横线，且长度 3-50
        if not re.fullmatch(r'^[A-Za-z0-9_-]{3,50}$', v):
            raise ValueError('Metadata name must be 3-50 characters long and contain only letters, numbers, underscores, and hyphens.')
        return v

    @field_validator("value_type")
    def validate_metadata_value_type(cls, v):
        if v not in [MetadataValueType.STRING, MetadataValueType.NUMBER, MetadataValueType.DATETIME]:
            raise ValueError('Metadata value type must be string, number, or datetime.')
        return v



class KbMetadataEntity(SQLModel, table=True):
    __tablename__ = "pai_knowledgebase_metadata"
    __table_args__ = (UniqueConstraint("name", "kb_id", "tenant_id", name="unique_kb_metadata"),)

    # metadata id
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=3, max_length=50, primary_key=True)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    kb_id: str = Field(default=None, foreign_key="pai_knowledgebase.id", ondelete="CASCADE", max_length=64)

    name: str = Field(default=None, min_length=3, max_length=50)
    value_type: str = Field(default=MetadataValueType.STRING)
    description: str = Field(default=None, sa_column=Column(Text))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime)
    )

    @field_validator("name")
    def validate_metadata_name_format(cls, v):
        # 使用正则表达式检查是否只包含字母、数字、下划线和短横线，且长度 3-50
        if not re.fullmatch(r'^[A-Za-z0-9_-]{3,50}$', v):
            raise ValueError('Metadata name must be 3-50 characters long and contain only letters, numbers, underscores, and hyphens.')
        return v

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()

# 记录文件-元数据映射关系
class FileMetadataEntity(SQLModel, table=True):
    __tablename__ = "pai_file_metadata"
    __table_args__ = (UniqueConstraint("id", "kb_id", "file_id", "tenant_id", name="unique_kb_file_metadata"),)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=3, max_length=64, primary_key=True)
    kb_id: str = Field(default=None, foreign_key="pai_knowledgebase.id", ondelete="CASCADE", max_length=64)
    file_id: str = Field(default=None, foreign_key="pai_knowledgebase_file.id", ondelete="CASCADE", max_length=64)
    metadata_id: str = Field(default=None, foreign_key="pai_knowledgebase_metadata.id", ondelete="CASCADE", max_length=64)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime)
    )


    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()
