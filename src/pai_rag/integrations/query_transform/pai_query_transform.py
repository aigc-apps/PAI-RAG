from typing import List, Optional, Sequence
from llama_index.core.settings import Settings
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.base.llms.types import ChatMessage
from pai_rag.utils.prompt_template import (
    INTENT_REWRITE_PROMPT_ZH,
    CONDENSE_QUESTION_ANSWER_PROMPT_ZH,
    KNOWLEDGEBASE_REWRITE_PROMPT_ZH,
    CHAT_LLM_REWRITE_PROMPT_ZH,
    WEBSEARCH_REWRITE_PROMPT_ZH,
    NL2SQL_REWRITE_PROMPT_ZH,
    NEWS_REWRITE_PROMPT_ZH,
    AGENT_REWRITE_PROMPT_ZH,
    REWRITE_PROMPT_ROLE_ZH,
)
from pai_rag.integrations.synthesizer.prompt_templates import CURRENT_QUERY_TIME_PROMPT
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.prompts import PromptTemplate
from pai_rag.app.api.models import ChatToolType, ChatIntentType, PaiQueryBundle
from pai_rag.utils.json_parser import parse_json_from_code_block_str
from datetime import datetime
from loguru import logger
import re


query_rewrite_prompts = {
    ChatToolType.CHAT_LLM: CHAT_LLM_REWRITE_PROMPT_ZH,
    ChatToolType.CHAT_DB: NL2SQL_REWRITE_PROMPT_ZH,
    ChatToolType.CHAT_KNOWLEDGEBASE: KNOWLEDGEBASE_REWRITE_PROMPT_ZH,
    ChatToolType.SEARCH_WEB: WEBSEARCH_REWRITE_PROMPT_ZH,
    ChatToolType.CHAT_NEWS: NEWS_REWRITE_PROMPT_ZH,
    ChatToolType.CHAT_AGENT: AGENT_REWRITE_PROMPT_ZH,
}


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

        additional_kwargs = message.additional_kwargs
        if additional_kwargs:
            string_message += f"\n{additional_kwargs}"
        string_messages.append(string_message)

    return "\n".join(string_messages)


class OpenAICompatibleQueryTransform:
    def __init__(
        self,
        llm: Optional[LLMType] = None,
        query_transform_prompt: Optional[BasePromptTemplate] = None,
        condense_question_prompt: Optional[BasePromptTemplate] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__()

        self._llm = (
            resolve_llm(llm, callback_manager=callback_manager) if llm else Settings.llm
        )
        self._query_transform_prompt = (
            query_transform_prompt or INTENT_REWRITE_PROMPT_ZH
        )
        default_condense_question_prompt = PromptTemplate(
            template="{}\n{}\n{}".format(
                self._query_transform_prompt,
                CURRENT_QUERY_TIME_PROMPT.format(
                    current_datetime=datetime.now().strftime("%Y年%m月%d日")
                ),
                CONDENSE_QUESTION_ANSWER_PROMPT_ZH,
            )
        )
        self._condense_question_prompt = (
            condense_question_prompt or default_condense_question_prompt
        )

    def get_prompt(self, query_str: str, chat_history: str, potential_intents):
        tool_prompt = "\n\n".join(
            [query_rewrite_prompts[intent] for intent in potential_intents]
        )
        return PromptTemplate(
            template=REWRITE_PROMPT_ROLE_ZH.format(
                tool_list=tool_prompt,
                chat_history=chat_history,
                query_str=query_str,
                cur_date=datetime.now().strftime("%Y年%m月%d日"),
            )
        )

    async def arun(
        self,
        chat_messages: List[ChatMessage] = [],
        potential_intents: List[ChatToolType] = [],
    ) -> PaiQueryBundle:
        chat_history_str = messages_to_history_str(
            chat_messages[-7:-1], max_length=1000
        )
        query_str = chat_messages[-1].content
        rewrite_prompt = self.get_prompt(
            chat_history=chat_history_str,
            query_str=query_str,
            potential_intents=potential_intents,
        )

        logger.debug(
            f"Chat history: {chat_history_str} \n rewrite_prompt: {rewrite_prompt}"
        )
        messages = self._llm._get_messages(
            rewrite_prompt,
            question=query_str,
            chat_history=chat_history_str,
        )
        chat_response = await self._llm.achat(
            messages=messages,
        )
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
        query = query_json.get("query", query_str)

        return PaiQueryBundle(
            intent=intent,
            messages=chat_messages,
            query_str=query,
            custom_embedding_strs=[transformed_query_str],
            chat_messages_str=chat_history_str,
            completion_tokens=chat_response.additional_kwargs.get(
                "completion_tokens", 0
            ),
            prompt_tokens=chat_response.additional_kwargs.get("prompt_tokens", 0),
            total_tokens=chat_response.additional_kwargs.get("total_tokens", 0),
        )
