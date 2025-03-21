from pydantic import BaseModel
from pai_rag.integrations.llms.pai.llm_config import (
    OpenAICompatibleLlmConfig,
)

DEFAULT_NEWS_ENDPOINT = "quanmiaolightapp.cn-beijing.aliyuncs.com"
DEFAULT_TOP_NEWS_COUNT = 10


class MiaobiNewsConfig(BaseModel):
    workspace_id: str | None = None
    access_key_id: str | None = None
    access_key_secret: str | None = None
    endpoint: str = DEFAULT_NEWS_ENDPOINT
    top_news_count: int = DEFAULT_TOP_NEWS_COUNT
    model_id: str | None = None
    llm: OpenAICompatibleLlmConfig | None = OpenAICompatibleLlmConfig()

    def is_enabled(self) -> bool:
        return (
            self.access_key_id is not None
            and self.access_key_secret is not None
            and self.workspace_id is not None
        )
