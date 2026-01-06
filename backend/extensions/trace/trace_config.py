from pydantic import BaseModel


class TraceConfig(BaseModel):
    exporter_type: str = "grpc"
    service_name: str | None = None
    token: str | None = None
    endpoint: str | None = None
    enabled: bool | None = None
    # user key to trace key mapping
    # example: {"vin": "gen_ai.user.id", }
    user_args: dict | None = None

    def is_enabled(self) -> bool:
        return self.enabled and self.service_name and self.endpoint
