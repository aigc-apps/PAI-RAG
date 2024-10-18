from enum import Enum
from typing import Dict
from pydantic import BaseModel
from llama_index.core.tools import ToolMetadata
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.output_parsers.selection import SelectionOutputParser
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import QueryType

DEFAULT_WEBSEARCH_DESCRIPTION = """
This tool is to help you get information from web.
It's useful for realtime news and common sense questions.
"""

DEFAULT_TOOL_DESCRIPTION = """
This tool can help you get travel information about weather, flights, train and hotels.
"""

DEFAULT_RAG_DESCRIPTION = """
This tool can help you get more specific information from the knowledge base.
"""


class Intents(str, Enum):
    WEBSEARCH = "websearch"
    RAG = "rag"
    TOOL = "tool"


class IntentConfig(BaseModel):
    descriptions: Dict[str, str] = {}


class PaiIntentRouter:
    def __init__(
        self,
        intent_config: IntentConfig,
        llm: LLM,
    ):
        if intent_config.descriptions is None or len(intent_config.descriptions) == 0:
            intent_config.descriptions = {
                Intents.RAG: DEFAULT_RAG_DESCRIPTION,
                Intents.TOOL: DEFAULT_TOOL_DESCRIPTION,
            }

        self.choices = [
            ToolMetadata(name=name, description=description)
            for name, description in intent_config.descriptions.items()
        ]
        self.selector = LLMSingleSelector.from_defaults(
            llm=llm, output_parser=SelectionOutputParser()
        )

    async def aselect(self, str_or_query_bundle: QueryType) -> Intents:
        if len(self.choices) == 1:
            return self.choices[0].name

        selector_result = await self.selector.aselect(
            choices=self.choices, query=str_or_query_bundle
        )
        assert (
            selector_result.selections > 0
        ), f"intent detection failed. {selector_result}"
        select_index = selector_result.selections[0].index
        return self.choices[select_index].name
