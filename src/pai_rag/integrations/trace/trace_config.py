from pydantic import BaseModel


class TraceConfig(BaseModel):
    service_name: str | None = None
    token: str | None = None
    endpoint: str | None = None

    def is_enabled(self) -> bool:
        return (
            self.service_name
            and self.token
            and self.endpoint
        )
    