from sqlmodel import Field, SQLModel


class TraceModel(SQLModel):
    endpoint: str = Field(default="http://tracing-analysis-dc-hz.aliyuncs.com:8090")
    token: str = Field(default=None)
    service_name: str = Field(default=None)
    enabled: bool = Field(default=False)


class TraceModelEntity(TraceModel, table=True):
    __tablename__ = "pai_trace_config"

    id: str = Field(default="default_trace_id", primary_key=True)
