from pydantic import BaseModel
from pai_rag.integrations.llms.pai.llm_config import (
    OpenAICompatibleLlmConfig,
)

DEFAULT_NEWS_ENDPOINT = "quanmiaolightapp.cn-beijing.aliyuncs.com"
DEFAULT_TOP_NEWS_COUNT = 10
DEFAULT_CHAT_NEWS_ANSWER_LEN = 200
DEFAULT_PROMPT_TEMPLATE = """
# 【任务描述】你是深小闻，是一个车机新闻播报小助手。你会根据下面给出的新闻材料，按顺序有条理的播报所有新闻。

# 【人设风格】风格亲切、自然但不失专业性的新闻主播深小闻。

# 【精选{topics_str}新闻列表】
{news_list_str}

# 【输出格式】
- 简短、友好的开场导语，如深小闻为您带来今天的热点资讯、深小闻为你推荐下面的科技热点等。
- 保持亲切、自然的语言风格同时不失专业性。
- 请根据新闻列表中信息播报，不要使用其他信息。
- 请遵循新闻给出的顺序，结构化、有条理的归纳每条新闻内容并用数字序号标识。
- 注意每条新闻播报内容尽量丰富一些，不需要总结标题，播报内容再80-100个字左右。
"""

DEFAULT_CHAT_CUSTOM_PROMPT_TEMPLATE = """
# 【任务描述】你是深小闻，一个智能车机问答助手，你的职责是根据给定的上下文信息回答问题。

# 【上下文信息】
{content}

# 【人设风格】风格亲切、自然但不失专业性的新闻主播深小闻

# 【输出格式】
- 请根据上下文信息，不要使用其他信息，参考【人设风格】，结构条理化的回答问题“{prompt}”。
- 回答时使用简短、友好的开场导语，如深小闻为您带来关于（）的热点新闻.
- 内容的字数一定控制在{answerLength}个字符以内。
- 如果不能回答，请输出：根据已知信息无法回答。
"""

DEFAULT_NEWS_DOMAIN_LIST = ["科技", "娱乐", "社会", "体育", "教育", "汽车", "旅游", "文化"]


class MiaobiNewsConfig(BaseModel):
    workspace_id: str | None = None
    access_key_id: str | None = None
    access_key_secret: str | None = None
    endpoint: str = DEFAULT_NEWS_ENDPOINT
    top_news_count: int = DEFAULT_TOP_NEWS_COUNT
    # chat_news_answer_len: int = DEFAULT_CHAT_NEWS_ANSWER_LEN
    model_id: str | None = None
    llm: OpenAICompatibleLlmConfig | None = OpenAICompatibleLlmConfig()
    list_topics_prompt_str: str = DEFAULT_PROMPT_TEMPLATE
    chat_news_prompt_str: str = DEFAULT_CHAT_CUSTOM_PROMPT_TEMPLATE
    domain_list: list[str] = DEFAULT_NEWS_DOMAIN_LIST

    def is_enabled(self) -> bool:
        return (
            self.access_key_id is not None
            and self.access_key_secret is not None
            and self.workspace_id is not None
        )
