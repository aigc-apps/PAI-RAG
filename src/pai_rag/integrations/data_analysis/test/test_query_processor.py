import os
from dotenv import load_dotenv

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.schema import QueryBundle
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.core import Settings

from pai_rag.integrations.data_analysis.text2sql.query_processor import KeywordExtractor
from pai_rag.integrations.llms.pai.llm_config import (
    DashScopeLlmConfig,
)


# 加载 .env 文件中的环境变量
load_dotenv()

llm_ds = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.1,
    max_tokens=2048,
)
print("DashScope:", llm_ds.metadata.is_function_calling_model)

llm_config = DashScopeLlmConfig()

llm_ol = OpenAILike(
    model=llm_config.model,
    api_base=llm_config.base_url,
    temperature=llm_config.temperature,
    system_prompt=llm_config.system_prompt,
    is_chat_model=True,
    api_key=llm_config.api_key or os.environ.get("DASHSCOPE_API_KEY"),
    max_tokens=llm_config.max_tokens,
    reuse_client=False,
    is_function_calling_model=True,
)

Settings.llm = llm_ol
query = "有猫的学生有多少？"
qp = KeywordExtractor()
result = qp.process(QueryBundle(query))
print(result)


# # def test_query_processor():
# #     query = "有猫的学生有多少？"
# #     qp = KeywordExtractor()
# #     keywords = qp.process(QueryBundle(query))
# #     assert isinstance(keywords, list)
# #     assert len(keywords) > 0
# #     assert "猫" in keywords
# #     assert "学生" in keywords
