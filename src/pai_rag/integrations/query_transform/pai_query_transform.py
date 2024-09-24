from abc import abstractmethod
from typing import Dict, List, Optional, cast
from llama_index.core.settings import Settings
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle, QueryType
from llama_index.core.base.llms.generic_utils import messages_to_history_str
from llama_index.core.storage.chat_store.base import BaseChatStore

from pai_rag.utils.prompt_template import (
    CONDENSE_QUESTION_CHAT_ENGINE_PROMPT_ZH,
    DEFAULT_FUSION_TRANSFORM_PROMPT,
)

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.prompts import PromptTemplate
from pai_rag.utils.messages_utils import parse_chat_messages

DEFAULT_FUSION_NUM_QUERIES = 4


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
            print(f"Generated queries:\n{queries_str}")

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
            print(f"Generated queries:\n{queries_str}")

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


class PaiCondenseQueryTransform(PaiBaseQueryTransform):
    def __init__(
        self,
        chat_store: BaseChatStore = None,
        llm: Optional[LLMType] = None,
        condense_question_prompt: Optional[BasePromptTemplate] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__()

        self._llm = (
            resolve_llm(llm, callback_manager=callback_manager) if llm else Settings.llm
        )
        self._condense_question_prompt = (
            condense_question_prompt or CONDENSE_QUESTION_CHAT_ENGINE_PROMPT_ZH
        )
        self._chat_store = chat_store

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"condense_query_prompt": PromptTemplate(self._condense_question_prompt)}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "condense_query_prompt" in prompts:
            self._prompt = cast(
                PromptTemplate, prompts["condense_query_prompt"]
            ).template

    def _run(self, query_bundle: QueryBundle, session_id, chat_history) -> QueryBundle:
        """Run query transform.
        Generate standalone question from conversation context and last message."""
        query_str = query_bundle.query_str

        if chat_history is not None:
            history_messages = parse_chat_messages(chat_history)
            for hist_mes in history_messages:
                self._chat_store.add_message(hist_mes)

        chat_history = self._chat_store.get_messages(session_id)
        if not chat_history:
            # Keep the question as is if there's no conversation context.
            return query_bundle

        chat_history_str = messages_to_history_str(chat_history)

        query_bundle_str = self._llm.predict(
            self._condense_question_prompt,
            question=query_str,
            chat_history=chat_history_str,
        )

        return QueryBundle(
            query_str=query_bundle_str,
            custom_embedding_strs=[query_bundle_str],
        )

    def run(
        self,
        query_bundle_or_str: QueryType,
        session_id: str | None = None,
        chat_history: List[Dict[str, str]] | None = None,
    ) -> QueryBundle:
        """Run query transform."""
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(
                query_str=query_bundle_or_str,
                custom_embedding_strs=[query_bundle_or_str],
            )
        else:
            query_bundle = query_bundle_or_str

        return self._run(query_bundle, session_id=session_id, chat_history=chat_history)

    async def _arun(
        self, query_bundle: QueryBundle, session_id, chat_history
    ) -> QueryBundle:
        """Run query transform.
        Generate standalone question from conversation context and last message."""
        query_str = query_bundle.query_str

        if chat_history is not None:
            history_messages = parse_chat_messages(chat_history)
            for hist_mes in history_messages:
                self._chat_store.add_message(hist_mes)

        chat_history = self._chat_store.get_messages(session_id)
        if not chat_history:
            # Keep the question as is if there's no conversation context.
            return query_bundle

        chat_history_str = messages_to_history_str(chat_history)
        query_bundle_str = await self._llm.apredict(
            self._condense_question_prompt,
            question=query_str,
            chat_history=chat_history_str,
        )

        return QueryBundle(
            query_str=query_bundle_str,
            custom_embedding_strs=[query_bundle_str],
        )

    async def arun(
        self,
        query_bundle_or_str: QueryType,
        session_id: str | None = None,
        chat_history: List[Dict[str, str]] | None = None,
    ) -> QueryBundle:
        """Run query transform."""
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(
                query_str=query_bundle_or_str,
                custom_embedding_strs=[query_bundle_or_str],
            )
        else:
            query_bundle = query_bundle_or_str

        return await self._arun(
            query_bundle, session_id=session_id, chat_history=chat_history
        )
