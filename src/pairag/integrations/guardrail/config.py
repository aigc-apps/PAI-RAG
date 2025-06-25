from pydantic import BaseModel

DEFAULT_GUARDRAIL_ADVICE = "作为人工智能助手，我无法回应包含不当或敏感信息的内容。"


class AliyunTextModerationPlusConfig(BaseModel):
    endpoint: str | None = None
    region: str | None = None
    access_key_id: str | None = None
    access_key_secret: str | None = None
    custom_advice: str | None = DEFAULT_GUARDRAIL_ADVICE

    def is_enabled(self) -> bool:
        return (
            self.access_key_id is not None
            and self.access_key_secret is not None
            and len(self.access_key_id) > 0
            and len(self.access_key_secret) > 0
            and self.endpoint is not None
            and self.endpoint != ""
            and self.region is not None
            and self.region != ""
        )
