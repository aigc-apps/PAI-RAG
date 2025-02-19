from enum import Enum
from typing import Dict
from pydantic import BaseModel
from llama_index.core.tools import ToolMetadata
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.output_parsers.selection import SelectionOutputParser
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import QueryType

# DEFAULT_WEBSEARCH_DESCRIPTION = """
# This tool is to help you get information from web.
# It's useful for realtime news and common sense questions.
# """

DEFAULT_TOOL_DESCRIPTION = """
This tool can help you get travel information about time, weather, flights, train and hotels.
"""

DEFAULT_RAG_DESCRIPTION = """
This tool can help you get more specific information from the knowledge base.
"""

DEFAULT_CHAT_DESCRIPTION = """
用于处理常规对话和互动，适用于日常交流、讨论、情感支持。不需要访问实时网络信息。
"""

DEFAULT_WEBSEARCH_DESCRIPTION = """
用于需要获取最新信息、查找具体数据或进行实时在线搜索以回答用户问题的场景。适合处理涉及当前时间、当前日期、当前事件、统计数据、特定事实或需要访问外部资源的查询，以确保提供最新和准确的回答。
"""


class Intents(str, Enum):
    WEBSEARCH = "websearch"
    RAG = "rag"
    TOOL = "tool"
    NL2SQL = "nl2sql"
    CHAT = "chat"


DEFAULT_INTENT_DESCRIPTIONS = {
    Intents.RAG: DEFAULT_RAG_DESCRIPTION,
    Intents.TOOL: DEFAULT_TOOL_DESCRIPTION,
}

DEFAULT_WEBSEARCH_DESCRIPTIONS = {
    Intents.WEBSEARCH: DEFAULT_WEBSEARCH_DESCRIPTION,
    Intents.CHAT: DEFAULT_CHAT_DESCRIPTION,
}


class IntentConfig(BaseModel):
    descriptions: Dict[Intents, str] = DEFAULT_INTENT_DESCRIPTIONS


class PaiIntentRouter:
    def __init__(
        self,
        intent_config: IntentConfig,
        llm: LLM,
    ):
        self.choices = [
            ToolMetadata(name=name, description=description)
            for name, description in intent_config.descriptions.items()
        ]
        self.selector = LLMSingleSelector.from_defaults(
            llm=llm, output_parser=SelectionOutputParser()
        )

    async def aselect(self, str_or_query_bundle: QueryType) -> Intents:
        if len(self.choices) <= 0:
            return Intents.RAG
        elif len(self.choices) == 1:
            return self.choices[0].name

        selector_result = await self.selector.aselect(
            choices=self.choices, query=str_or_query_bundle
        )
        assert (
            len(selector_result.selections) > 0
        ), f"intent detection failed. {selector_result}"
        select_index = selector_result.selections[0].index
        return self.choices[select_index].name

    def select(self, str_or_query_bundle: QueryType) -> Intents:
        if len(self.choices) <= 1:
            return Intents.RAG

        selector_result = self.selector.select(
            choices=self.choices, query=str_or_query_bundle
        )
        assert (
            len(selector_result.selections) > 0
        ), f"intent detection failed. {selector_result}"
        select_index = selector_result.selections[0].index
        return self.choices[select_index].name
