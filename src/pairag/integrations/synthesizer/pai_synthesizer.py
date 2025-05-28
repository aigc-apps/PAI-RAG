from typing import Any, List, Dict, Optional, Sequence, Union

from llama_index.core.settings import Settings
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms import LLM
from llama_index.core.types import RESPONSE_TEXT_TYPE
import llama_index.core.instrumentation as instrument
from llama_index.core.schema import (
    NodeWithScore,
    QueryType,
)
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
)
from llama_index.core.instrumentation.events.synthesis import (
    SynthesizeStartEvent,
)
from llama_index.core.base.llms.types import ChatResponse, ChatResponseAsyncGen
from llama_index.core.prompts import PromptTemplate
from pairag.chat.models import ChatResponseWrapper
from pairag.integrations.synthesizer.prompt_templates import (
    DEFAULT_SYSTEM_ROLE_TEMPLATE,
    DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    DEFAULT_CONTEXT_ANSWER_TEMPLATE,
    CURRENT_TIME_PROMPT,
)
from loguru import logger

from pairag.utils.time_utils import get_prompt_current_time_str

dispatcher = instrument.get_dispatcher(__name__)


"""
PaiSynthesizer:
Supports multi-modal inputs synthesizer.
Will use Multi-modal LLM for inputs with images and LLM for pure text inputs.
"""


class PaiSynthesizer:
    def __init__(
        self,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
    ) -> None:
        self._llm = llm
        self._callback_manager = callback_manager or Settings.callback_manager
        self._prompt_helper = (
            prompt_helper
            or Settings._prompt_helper
            or PromptHelper.from_llm_metadata(
                self._llm.metadata,
            )
        )

    @property
    def callback_manager(self) -> CallbackManager:
        return self._callback_manager

    @callback_manager.setter
    def callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self._callback_manager = callback_manager
        # TODO: please fix this later
        self._callback_manager = callback_manager
        self._llm.callback_manager = callback_manager

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "llm_only_template": self._llm_only_template,
            "text_qa_template": self._text_qa_template,
            "citation_template": self._citation_text_qa_template,
            "multimodal_qa_template": self._multimodal_qa_template,
            "citation_multimodal_qa_template": self._citation_multimodal_qa_template,
        }

    @dispatcher.span
    def synthesize(
        self,
        query: QueryType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        system_role_str: str = None,
        prompt_template_str: str = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        raise NotImplementedError

    @dispatcher.span
    async def asynthesize(
        self,
        query_str: str,
        chat_history_str: str,
        nodes: List[NodeWithScore],
        stream: bool = False,
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        system_role_str: str = DEFAULT_SYSTEM_ROLE_TEMPLATE,
        prompt_template_str: str = DEFAULT_CUSTOM_PROMPT_TEMPLATE,
        prompt_template_args: Dict[str, str] = None,
        **response_kwargs: Any,
    ) -> ChatResponseWrapper:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query_str,
            )
        )

        with self.callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query_str},
        ) as event:
            response = await self.aget_response(
                query_str=query_str,
                nodes=nodes,
                history_str=chat_history_str,
                streaming=stream,
                system_role_str=system_role_str,
                prompt_template_str=prompt_template_str,
                prompt_template_args=prompt_template_args or {},
                **response_kwargs,
            )
            additional_source_nodes = additional_source_nodes or []
            source_nodes = list(nodes) + list(additional_source_nodes)
            event.on_end(payload={EventPayload.RESPONSE: response})

        return ChatResponseWrapper(response=response, source_nodes=source_nodes)

    def _contruct_context_str(
        self,
        nodes: List[NodeWithScore],
    ):
        context_str = ""
        for i, node in enumerate(nodes):
            context_str += f"""
材料 {i+1}:
{node.node.get_content()}

                """
        return context_str

    async def aget_response(
        self,
        query_str: str,
        nodes: List[NodeWithScore],
        history_str: str = None,
        streaming: bool = False,
        citation: bool = False,
        system_role_str: str = None,
        prompt_template_str: str = None,
        prompt_template_args: Dict[str, str] = None,
        **response_kwargs: Any,
    ) -> Union[ChatResponse, ChatResponseAsyncGen]:
        context_str = self._contruct_context_str(nodes)
        cur_date = get_prompt_current_time_str()
        logger.info(f"Synthesize using LLM with  citation flag: {citation}")
        prompt_template = PromptTemplate(
            template="{}\n{}\n{}\n{}".format(
                system_role_str,
                prompt_template_str,
                CURRENT_TIME_PROMPT.format(current_datetime=cur_date),
                DEFAULT_CONTEXT_ANSWER_TEMPLATE,
            )
        )

        prompt_template_args.update(
            {"query_str": query_str, "history_str": history_str}
        )
        text_qa_template = prompt_template.partial_format(**prompt_template_args)

        response: RESPONSE_TEXT_TYPE
        logger.info(
            f"Synthsize using LLM with contexts. \n Prompt: {text_qa_template} \n Chat History: {history_str} \n Query: {query_str}"
        )

        logger.info(f"Prompt_helper parameter: {str(self._prompt_helper)}")
        truncated_context_list = self._prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=[
                context_str
            ],  # 目前主要处理context_str一个参数，可以是：rag检索结果/web搜索结果/db查询结果
        )
        logger.info(f"Truncated_context_str: {str(truncated_context_list)}")

        messages = self._llm._get_messages(
            text_qa_template,
            context_str=truncated_context_list[0],
            **response_kwargs,
        )

        if not streaming:
            response = await self._llm.achat(
                messages=messages,
                **response_kwargs,
            )
        else:
            response = await self._llm.astream_chat(
                messages=messages,
                **response_kwargs,
            )

        return response

    def get_response(self, query_str, text_chunks, **response_kwargs):
        raise NotImplementedError
