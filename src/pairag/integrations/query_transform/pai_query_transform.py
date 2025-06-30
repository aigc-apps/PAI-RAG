from typing import List, Optional, Sequence
from llama_index.core.llms.utils import LLMType
from llama_index.core.base.llms.types import ChatMessage
from pairag.integrations.query_transform.intent_models import (
    ChatIntentType,
    ChatToolType,
    IntentResult,
)
from pairag.utils.prompt_template import (
    KNOWLEDGEBASE_REWRITE_PROMPT_ZH,
    CHAT_LLM_REWRITE_PROMPT_ZH,
    WEBSEARCH_REWRITE_PROMPT_ZH,
    NL2SQL_REWRITE_PROMPT_ZH,
    NEWS_REWRITE_PROMPT_ZH,
    REWRITE_PROMPT_ROLE_ZH,
)
from llama_index.core.prompts import PromptTemplate
from pairag.utils.json_parser import parse_json_from_code_block_str
from loguru import logger
import re
from openai.types.completion_usage import CompletionUsage
from pairag.utils.time_utils import get_prompt_current_time_str


def messages_to_history_str(
    messages: Sequence[ChatMessage], max_length: int = 1000
) -> str:
    """Convert messages to a history string."""
    string_messages = []
    for message in messages:
        if not message.content:
            continue

        role = message.role

        content = message.content[:max_length]

        string_message = f"{role.value}: {content}"

        string_messages.append(string_message)

    return "\n".join(string_messages)


def check_keywords_in_string(string, keywords):
    # 使用 any() 函数检查是否有任何一个关键字出现在字符串中
    return any(keyword in string for keyword in keywords)


class OpenAICompatibleQueryTransform:
    def __init__(
        self,
        llm: Optional[LLMType] = None,
        base_transform_prompt: str = REWRITE_PROMPT_ROLE_ZH,
        llm_tool_prompt_str: str = CHAT_LLM_REWRITE_PROMPT_ZH,
        knowledge_tool_prompt_str: str = KNOWLEDGEBASE_REWRITE_PROMPT_ZH,
        websearch_tool_prompt_str: str = WEBSEARCH_REWRITE_PROMPT_ZH,
        db_tool_prompt_str: str = NL2SQL_REWRITE_PROMPT_ZH,
        news_tool_prompt_str: str = NEWS_REWRITE_PROMPT_ZH,
    ):
        super().__init__()

        self._llm = llm
        self._base_transform_prompt = PromptTemplate(template=base_transform_prompt)

        self._tool_prompts = {
            ChatToolType.CHAT_LLM: llm_tool_prompt_str,
            ChatToolType.CHAT_DB: db_tool_prompt_str,
            ChatToolType.CHAT_KNOWLEDGEBASE: knowledge_tool_prompt_str,
            ChatToolType.SEARCH_WEB: websearch_tool_prompt_str,
            ChatToolType.CHAT_NEWS: news_tool_prompt_str,
        }

    def get_prompt(self, query_str: str, chat_history: str, potential_intents):
        tool_prompt = "\n\n".join(
            [self._tool_prompts[intent] for intent in potential_intents]
        )
        current_datetime = get_prompt_current_time_str()
        return PromptTemplate(
            template=self._base_transform_prompt.format(
                tool_list=tool_prompt,
                chat_history=chat_history,
                query_str=query_str,
                cur_date=current_datetime,  # backward compatibility
                current_datetime=current_datetime,
            )
        )

    async def arun(
        self,
        chat_messages: List[ChatMessage] = [],
        chat_history_str: str = None,
        potential_intents: List[ChatToolType] = [],
    ) -> IntentResult:
        query_str = chat_messages[-1].content
        rewrite_prompt = self.get_prompt(
            chat_history=chat_history_str,
            query_str=query_str,
            potential_intents=potential_intents,
        )

        logger.debug(
            f"Chat history: {chat_history_str} \n rewrite_prompt: {rewrite_prompt}"
        )
        messages = self._llm._get_messages(rewrite_prompt)
        chat_response = await self._llm.achat(messages=messages)

        transformed_query_str = chat_response.message.content

        logger.debug(f"Transformed query [{query_str}] --> [{transformed_query_str}]")
        # 修复thought输出
        transformed_query_str = re.sub(
            r"<think>.*?</think>\n*", "", transformed_query_str, flags=re.DOTALL
        )
        transformed_query_str = transformed_query_str.replace("<think>", "").replace(
            "</think>", ""
        )
        query_json = parse_json_from_code_block_str(transformed_query_str)
        intent = query_json.get("intent", ChatIntentType.CHAT_KNOWLEDGEBASE)
        new_query_str = query_json.get("query", query_str)

        return IntentResult(
            intent=intent,
            query_str=new_query_str,
            token_usage=CompletionUsage(
                prompt_tokens=chat_response.additional_kwargs.get("prompt_tokens", 0),
                completion_tokens=chat_response.additional_kwargs.get(
                    "completion_tokens", 0
                ),
                total_tokens=chat_response.additional_kwargs.get("total_tokens", 0),
            ),
        )
