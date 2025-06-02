from pydantic import BaseModel


class TraceConfig(BaseModel):
    service_name: str | None = None
    token: str | None = None
    endpoint: str | None = None
    enabled: bool | None = None

    def is_enabled(self) -> bool:
        return self.enabled and self.service_name and self.token and self.endpoint
