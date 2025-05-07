from typing import Any, Generator, List, Dict, Optional, Sequence, Union, cast

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.llms import LLM
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.multi_modal_llms import MultiModalLLM
import llama_index.core.instrumentation as instrument
from llama_index.core.schema import (
    NodeWithScore,
    QueryBundle,
    QueryType,
)
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
)
from llama_index.core.instrumentation.events.synthesis import (
    SynthesizeStartEvent,
)
from llama_index.core.llms.llm import (
    astream_completion_response_to_tokens,
)
from llama_index.core.base.llms.types import ChatResponse, ChatResponseAsyncGen
from llama_index.core.prompts import PromptTemplate
from pai_rag.app.api.models import ChatResponseWrapper, PaiQueryBundle
from pai_rag.integrations.synthesizer.prompt_templates import (
    DEFAULT_EMPTY_RESPONSE_GEN,
    DEFAULT_SYSTEM_ROLE_TEMPLATE,
    DEFAULT_CUSTOM_PROMPT_TEMPLATE,
    DEFAULT_ANSWER_TEMPLATE,
    DEFAULT_CONTEXT_ANSWER_TEMPLATE,
    DEFAULT_CUSTOM_CITATION_PROMPR_TEMPLATE,
    CURRENT_TIME_PROMPT,
)
from pai_rag.app.web.ui_constants import SYN_GENERAL_PROMPTS
from loguru import logger

from pai_rag.utils.time_utils import get_prompt_current_time_str

dispatcher = instrument.get_dispatcher(__name__)

QueryTextType = QueryType

"""
PaiSynthesizer:
Supports multi-modal inputs synthesizer.
Will use Multi-modal LLM for inputs with images and LLM for pure text inputs.
"""


class PaiSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        system_role_template: Optional[str] = None,
        custom_prompt_template: Optional[str] = None,
        multimodal_llm: Optional[MultiModalLLM] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            streaming=streaming,
        )
        self._multimodal_llm = multimodal_llm
        self._update_prompts(
            system_role_str=system_role_template,
            prompt_template_str=custom_prompt_template,
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "llm_only_template": self._llm_only_template,
            "text_qa_template": self._text_qa_template,
            "citation_template": self._citation_text_qa_template,
            "multimodal_qa_template": self._multimodal_qa_template,
            "citation_multimodal_qa_template": self._citation_multimodal_qa_template,
        }

    def _update_prompts(
        self, system_role_str: str = None, prompt_template_str: str = None
    ) -> None:
        """Update prompts."""
        self._system_role_template = system_role_str or DEFAULT_SYSTEM_ROLE_TEMPLATE
        self._custom_prompt_template = (
            prompt_template_str or DEFAULT_CUSTOM_PROMPT_TEMPLATE
        )

        self._llm_only_template = PromptTemplate(
            template="{}\n{}\n{}\n{}".format(
                self._system_role_template,
                self._custom_prompt_template,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=get_prompt_current_time_str()
                ),
                DEFAULT_ANSWER_TEMPLATE,
            )
        )
        self._text_qa_template = PromptTemplate(
            template="{}\n{}\n{}\n{}".format(
                self._system_role_template,
                self._custom_prompt_template,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=get_prompt_current_time_str()
                ),
                DEFAULT_CONTEXT_ANSWER_TEMPLATE,
            )
        )
        self._citation_text_qa_template = PromptTemplate(
            template="{}\n{}\n{}\n{}\n{}".format(
                self._system_role_template,
                self._custom_prompt_template,
                DEFAULT_CUSTOM_CITATION_PROMPR_TEMPLATE,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=get_prompt_current_time_str()
                ),
                DEFAULT_CONTEXT_ANSWER_TEMPLATE,
            )
        )
        self._multimodal_qa_template = PromptTemplate(
            template="{}\n{}\n{}\n{}".format(
                self._system_role_template,
                self._custom_prompt_template,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=get_prompt_current_time_str()
                ),
                DEFAULT_CONTEXT_ANSWER_TEMPLATE,
            )
        )
        self._citation_multimodal_qa_template = PromptTemplate(
            template="{}\n{}\n{}\n{}\n{}".format(
                self._system_role_template,
                self._custom_prompt_template,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=get_prompt_current_time_str()
                ),
                DEFAULT_CUSTOM_CITATION_PROMPR_TEMPLATE,
                DEFAULT_CONTEXT_ANSWER_TEMPLATE,
            )
        )

    @dispatcher.span
    def synthesize(
        self,
        query: PaiQueryBundle,
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
        query: PaiQueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        system_role_str: str = None,
        prompt_template_str: str = None,
        prompt_template_args: Dict[str, str] = None,
        **response_kwargs: Any,
    ) -> ChatResponseWrapper:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
        ) as event:
            query_str = query.query_str

            if query.chat_messages_str:
                history_str = query.chat_messages_str
            else:
                history_str = ""
            if query.no_retrieval:
                response = await self.aget_llm_only_response(
                    query_str=query_str,
                    history_str=history_str,
                    streaming=query.stream,
                    system_role_str=system_role_str or self._system_role_template,
                    prompt_template_str=prompt_template_str
                    or self._custom_prompt_template,
                    **response_kwargs,
                )
            else:
                response = await self.aget_response(
                    query_str=query_str,
                    original_query_str=query.original_query_str,
                    nodes=nodes,
                    history_str=history_str,
                    streaming=query.stream,
                    citation=query.citation,
                    system_role_str=system_role_str or self._system_role_template,
                    prompt_template_str=prompt_template_str
                    or self._custom_prompt_template,
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
        original_query_str: str,
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
        if not citation:
            if prompt_template_args:
                prompt_template = PromptTemplate(
                    template="{}\n{}\n{}".format(
                        system_role_str,
                        CURRENT_TIME_PROMPT.format(current_datetime=cur_date),
                        SYN_GENERAL_PROMPTS,
                    )
                )
            else:
                prompt_template = (
                    PromptTemplate(
                        template="{}\n{}\n{}\n{}".format(
                            system_role_str,
                            prompt_template_str,
                            CURRENT_TIME_PROMPT.format(current_datetime=cur_date),
                            DEFAULT_CONTEXT_ANSWER_TEMPLATE,
                        )
                    )
                    or self._multimodal_qa_template
                )
        else:
            prompt_template = (
                PromptTemplate(
                    template="{}\n{}\n{}\n{}\n{}".format(
                        system_role_str,
                        prompt_template_str,
                        DEFAULT_CUSTOM_CITATION_PROMPR_TEMPLATE,
                        CURRENT_TIME_PROMPT.format(current_datetime=cur_date),
                        DEFAULT_CONTEXT_ANSWER_TEMPLATE,
                    )
                )
                or self._citation_multimodal_qa_template
            )

        if prompt_template_args:
            db_description_str = prompt_template_args.get("db_description_str", "")
            query_code_instruction = (
                [n.node.metadata["query_code_instruction"] for n in nodes],
            )
            text_qa_template = prompt_template.partial_format(
                query_str=query_str,
                db_schema=db_description_str,
                query_code_instruction=query_code_instruction,
            )
        else:
            text_qa_template = prompt_template.partial_format(
                history_str=history_str, query_str=query_str
            )

        response: RESPONSE_TEXT_TYPE
        logger.info(
            f"Synthsize using LLM with contexts. \n Prompt: {text_qa_template} \n Chat History: {history_str} \n Query: {query_str}"
        )

        logger.info(f"Prompt_helper parameter: {str(self._prompt_helper)}")
        truncated_context_str = self._prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=[context_str],
        )
        logger.info(f"Truncated_context_str: {str(truncated_context_str)}")

        messages = self._llm._get_messages(
            text_qa_template,
            context_str=truncated_context_str,
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

    async def aget_llm_only_response(
        self,
        query_str: str,
        history_str: str = None,
        streaming: bool = False,
        system_role_str: str = None,
        prompt_template_str: str = None,
        **kwargs: Any,
    ) -> Union[ChatResponse, ChatResponseAsyncGen]:
        response: RESPONSE_TEXT_TYPE
        _llm_only_template = PromptTemplate(
            template="{}\n{}\n{}\n{}".format(
                system_role_str,
                prompt_template_str,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=get_prompt_current_time_str()
                ),
                DEFAULT_ANSWER_TEMPLATE,
            )
        )

        _llm_only_template = _llm_only_template.partial_format(history_str=history_str)
        logger.info(
            f"Synthsize using LLM only. \n Prompt: {_llm_only_template}. \n Chat History: {history_str} \n Query: {query_str}"
        )
        messages = self._llm._get_messages(
            _llm_only_template,
            query_str=query_str,
            **kwargs,
        )

        if not streaming:
            response = await self._llm.achat(
                messages=messages,
                **kwargs,
            )
        else:
            response = await self._llm.astream_chat(
                messages=messages,
                **kwargs,
            )

        return response

    async def _aget_multi_modal_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
        citation: bool = False,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        image_documents = load_image_urls(image_url_list)

        context_str = (
            "\n".join([f"材料 {i+1}:\n{text}\n" for i, text in enumerate(text_chunks)])
            + "\n"
        )
        context_str += "\n".join(
            [f"图片 {i+1}:\n{url}\n" for i, url in enumerate(image_url_list)]
        )

        if not citation:
            fmt_prompt = self._multimodal_qa_template.format(
                context_str=context_str, query_str=query_str
            )
        else:
            fmt_prompt = self._citation_multimodal_qa_template.format(
                context_str=context_str, query_str=query_str
            )

        logger.info(
            f"Synthsize using Multi-modal LLM with fmt_prompt {fmt_prompt}. citation: {citation}"
        )
        if streaming:
            completion_response_gen = await self._multimodal_llm.astream_complete(
                prompt=fmt_prompt,
                image_documents=image_documents,
                **response_kwargs,
            )
            stream_tokens = await astream_completion_response_to_tokens(
                completion_response_gen
            )
            return cast(Generator, stream_tokens)
        else:
            llm_response = await self._multimodal_llm.acomplete(
                prompt=fmt_prompt,
                image_documents=image_documents,
                **response_kwargs,
            )
            response = llm_response.text or DEFAULT_EMPTY_RESPONSE_GEN
            return response

    def get_response(self, query_str, text_chunks, **response_kwargs):
        raise NotImplementedError
