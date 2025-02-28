from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, cast
from llama_index.core.settings import Settings
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle, QueryType
from llama_index.core.base.llms.types import ChatMessage
from pai_rag.utils.prompt_template import (
    CONDENSE_QUESTION_CHAT_ENGINE_PROMPT_ZH,
    DEFAULT_FUSION_TRANSFORM_PROMPT,
    CONDENSE_QUESTION_ANSWER_PROMPT_ZH,
)
from pai_rag.integrations.synthesizer.prompt_templates import CURRENT_TIME_PROMPT
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.prompts import PromptTemplate
from pai_rag.app.api.models import PaiQueryBundle
from pai_rag.utils.json_parser import parse_json_from_code_block_str
from datetime import datetime
from loguru import logger
import re

DEFAULT_FUSION_NUM_QUERIES = 4


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


class PaiBaseQueryTransform(BaseQueryTransform):
    @abstractmethod
    async def _arun(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform."""

    async def arun(
        self,
        query_bundle_or_str: QueryType,
        metadata: Optional[Dict] = None,
    ) -> QueryBundle:
        """Run query transform."""
        metadata = metadata or {}
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(
                query_str=query_bundle_or_str,
                custom_embedding_strs=[query_bundle_or_str],
            )
        else:
            query_bundle = query_bundle_or_str

        return await self._arun(query_bundle, metadata=metadata)


class PaiFusionQueryTransform(PaiBaseQueryTransform):
    def __init__(
        self,
        llm: Optional[LLMType] = None,
        fusion_transform_prompt: Optional[BasePromptTemplate] = None,
        num_queries: int = DEFAULT_FUSION_NUM_QUERIES,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """ """
        super().__init__()

        self._llm = (
            resolve_llm(llm, callback_manager=callback_manager) if llm else Settings.llm
        )
        self._prompt = fusion_transform_prompt or DEFAULT_FUSION_TRANSFORM_PROMPT
        self._num_queries = num_queries

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"fusion_query_prompt": PromptTemplate(self._prompt)}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "fusion_query_prompt" in prompts:
            self._prompt = cast(PromptTemplate, prompts["fusion_query_prompt"]).template

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> List[QueryBundle]:
        """Run query transform."""
        query_str = query_bundle.query_str
        prompt_str = self._prompt.format(
            num_queries=self._num_queries - 1,
            query=query_str,
        )
        response = self._llm.complete(prompt_str)

        # assume LLM proper put each query on a newline
        # TODO: query改写的结构化输出
        queries = response.text.split("\n")
        queries = [q.strip() for q in queries if q.strip()]
        if self._verbose:
            queries_str = "\n".join(queries)
            logger.info(f"Generated queries:\n{queries_str}")

        # The LLM often returns more queries than we asked for, so trim the list.
        return [
            QueryBundle(
                query_str=new_query_str,
                custom_embedding_strs=[new_query_str],
            )
            for new_query_str in queries[: self.num_queries - 1]
        ]

    async def _arun(
        self, query_bundle: QueryBundle, metadata: Dict
    ) -> List[QueryBundle]:
        """Run query transform."""
        query_str = query_bundle.query_str
        prompt_str = self._prompt.format(
            num_queries=self._num_queries - 1,
            query=query_str,
        )
        response = await self._llm.acomplete(prompt_str)

        # assume LLM proper put each query on a newline
        queries = response.text.split("\n")
        queries = [q.strip() for q in queries if q.strip()]
        if self._verbose:
            queries_str = "\n".join(queries)
            logger.info(f"Generated queries:\n{queries_str}")

        # The LLM often returns more queries than we asked for, so trim the list.
        return [
            QueryBundle(
                query_str=new_query_str,
                custom_embedding_strs=[new_query_str],
            )
            for new_query_str in queries[: self.num_queries - 1]
        ]


class PaiHyDEQueryTransform(PaiBaseQueryTransform, HyDEQueryTransform):
    async def _arun(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform."""
        # TODO: support generating multiple hypothetical docs
        query_str = query_bundle.query_str
        hypothetical_doc = await self._llm.apredict(
            self._hyde_prompt, context_str=query_str
        )
        embedding_strs = [hypothetical_doc]
        if self._include_original:
            embedding_strs.extend(query_bundle.embedding_strs)
        return QueryBundle(
            query_str=query_str,
            custom_embedding_strs=embedding_strs,
        )


class OpenAICompatibleQueryTransform:
    def __init__(
        self,
        llm: Optional[LLMType] = None,
        query_transform_prompt: Optional[BasePromptTemplate] = None,
        condense_question_prompt: Optional[BasePromptTemplate] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__()

        self._llm = (
            resolve_llm(llm, callback_manager=callback_manager) if llm else Settings.llm
        )
        self._query_transform_prompt = (
            query_transform_prompt or CONDENSE_QUESTION_CHAT_ENGINE_PROMPT_ZH
        )
        default_condense_question_prompt = PromptTemplate(
            template="{}\n{}\n{}".format(
                self._query_transform_prompt,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
                ),
                CONDENSE_QUESTION_ANSWER_PROMPT_ZH,
            )
        )
        self._condense_question_prompt = (
            condense_question_prompt or default_condense_question_prompt
        )

    def run(
        self,
        chat_messages: List[ChatMessage] = [],
    ) -> QueryBundle:
        chat_history_str = messages_to_history_str(chat_messages[-7:], max_length=500)
        current_condense_question_prompt = PromptTemplate(
            template="{}\n{}\n{}".format(
                self._query_transform_prompt,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
                ),
                CONDENSE_QUESTION_ANSWER_PROMPT_ZH,
            )
        )
        logger.debug(
            f"Chat history: {chat_history_str} \n condense_question_prompt: {current_condense_question_prompt}"
        )
        messages = self._llm._get_messages(
            current_condense_question_prompt,
            question=chat_messages[-1].content,
            chat_history=chat_history_str,
        )
        chat_response = self._llm.chat(
            messages=messages,
        )
        transformed_query_str = chat_response.message.content
        logger.debug(
            f"Transformed query [{chat_messages[-1].content}] --> [{transformed_query_str}]"
        )
        # 修复thought输出
        transformed_query_str = re.sub(
            r"<think>.*?</think>\n*", "", transformed_query_str, flags=re.DOTALL
        )
        query_json = parse_json_from_code_block_str(transformed_query_str)
        if ("query" not in query_json) or (len(query_json["query"]) == 0):
            return PaiQueryBundle(
                query_str=chat_messages[-1].content,
                need_web_search=False,
                custom_embedding_strs=[
                    chat_messages[-1].content,
                    transformed_query_str,
                ],
                chat_messages_str=chat_history_str,
                completion_tokens=chat_response.additional_kwargs.get(
                    "completion_tokens", 0
                ),
                prompt_tokens=chat_response.additional_kwargs.get("prompt_tokens", 0),
                total_tokens=chat_response.additional_kwargs.get("total_tokens", 0),
            )
        else:
            return PaiQueryBundle(
                query_str=query_json["query"],
                need_web_search=True,
                custom_embedding_strs=[
                    chat_messages[-1].content,
                    transformed_query_str,
                ],
                chat_messages_str=chat_history_str,
                completion_tokens=chat_response.additional_kwargs.get(
                    "completion_tokens", 0
                ),
                prompt_tokens=chat_response.additional_kwargs.get("prompt_tokens", 0),
                total_tokens=chat_response.additional_kwargs.get("total_tokens", 0),
            )

    async def arun(
        self,
        chat_messages: List[ChatMessage] = [],
    ) -> QueryBundle:
        """Run query transform.
        Generate standalone question from conversation context and last message."""
        chat_history_str = messages_to_history_str(chat_messages[-7:], max_length=500)
        current_condense_question_prompt = PromptTemplate(
            template="{}\n{}\n{}".format(
                self._query_transform_prompt,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
                ),
                CONDENSE_QUESTION_ANSWER_PROMPT_ZH,
            )
        )
        logger.debug(
            f"Chat history: {chat_history_str} \n condense_question_prompt: {current_condense_question_prompt}"
        )
        messages = self._llm._get_messages(
            current_condense_question_prompt,
            question=chat_messages[-1].content,
            chat_history=chat_history_str,
        )
        chat_response = await self._llm.achat(
            messages=messages,
        )
        transformed_query_str = chat_response.message.content

        logger.debug(
            f"Transformed query [{chat_messages[-1].content}] --> [{transformed_query_str}]"
        )
        # 修复thought输出
        transformed_query_str = re.sub(
            r"<think>.*?</think>\n*", "", transformed_query_str, flags=re.DOTALL
        )
        query_json = parse_json_from_code_block_str(transformed_query_str)

        if ("query" not in query_json) or (len(query_json["query"]) == 0):
            return PaiQueryBundle(
                query_str=chat_messages[-1].content,
                need_web_search=False,
                custom_embedding_strs=[chat_messages[-1].content],
                chat_messages_str=chat_history_str,
                completion_tokens=chat_response.additional_kwargs.get(
                    "completion_tokens", 0
                ),
                prompt_tokens=chat_response.additional_kwargs.get("prompt_tokens", 0),
                total_tokens=chat_response.additional_kwargs.get("total_tokens", 0),
            )
        else:
            if chat_messages[-1].content != query_json["query"]:
                chat_history_str += f' {query_json["query"]}'
            return PaiQueryBundle(
                query_str=query_json["query"],
                need_web_search=True,
                custom_embedding_strs=[transformed_query_str],
                chat_messages_str=chat_history_str,
                completion_tokens=chat_response.additional_kwargs.get(
                    "completion_tokens", 0
                ),
                prompt_tokens=chat_response.additional_kwargs.get("prompt_tokens", 0),
                total_tokens=chat_response.additional_kwargs.get("total_tokens", 0),
            )
