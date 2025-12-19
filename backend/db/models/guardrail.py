from datetime import datetime, timezone
import uuid
from pydantic import field_serializer
from sqlmodel import Column, DateTime, Field, SQLModel
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional


class GuardrailConfig(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    region_name: str = Field(default=None)
    region_id: str = Field(default="cn-hangzhou")
    endpoint: str = Field(default=None)


class GuardrailConfigRead(GuardrailConfig):
    id: str = Field(default=None)


class GuardrailConfigCreate(GuardrailConfig):
    access_key_id: str = Field(default=None)
    access_key_secret: str = Field(default=None)


class GuardrailConfigEntity(GuardrailConfig, table=True):
    __tablename__ = "pai_guardrail_config"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4().hex), primary_key=True)
    encrypted_access_key_id: str = Field(default=None)
    encrypted_access_key_secret: str = Field(default=None)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()
