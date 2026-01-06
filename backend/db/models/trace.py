from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON
from typing import Optional
from common.system_constants import DEFAULT_TENANT_ID
import uuid


class TraceModel(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    endpoint: str = Field(default="http://tracing-analysis-dc-hz.aliyuncs.com:8090")
    token: str = Field(default=None)
    service_name: str = Field(default=None)
    enabled: bool = Field(default=False)
    user_args: Optional[dict] = Field(default=None, sa_column=Column(JSON))


class TraceModelEntity(TraceModel, table=True):
    __tablename__ = "pai_trace_config"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4().hex), primary_key=True)
    def is_enabled(self) -> bool:
        return self.enabled and self.service_name and self.endpoint
