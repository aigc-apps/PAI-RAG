from typing import Any, Generator, List, Optional, Sequence, Union, cast

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
    ImageNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    QueryType,
)
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
)
from llama_index.core.instrumentation.events.synthesis import (
    SynthesizeStartEvent,
    SynthesizeEndEvent,
)
from llama_index.core.llms.llm import (
    stream_completion_response_to_tokens,
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
from loguru import logger
from datetime import datetime

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
                CURRENT_TIME_PROMPT.format(
                    current_datetime=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
                ),
                self._custom_prompt_template,
                DEFAULT_ANSWER_TEMPLATE,
            )
        )
        self._text_qa_template = PromptTemplate(
            template="{}\n{}\n{}\n{}".format(
                self._system_role_template,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
                ),
                self._custom_prompt_template,
                DEFAULT_CONTEXT_ANSWER_TEMPLATE,
            )
        )
        self._citation_text_qa_template = PromptTemplate(
            template="{}\n{}\n{}\n{}\n{}".format(
                self._system_role_template,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
                ),
                self._custom_prompt_template,
                DEFAULT_CUSTOM_CITATION_PROMPR_TEMPLATE,
                DEFAULT_CONTEXT_ANSWER_TEMPLATE,
            )
        )
        self._multimodal_qa_template = self._text_qa_template
        self._citation_multimodal_qa_template = self._citation_text_qa_template

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
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        text_nodes, image_nodes = [], []
        for node in nodes:
            if isinstance(node.node, ImageNode):
                image_nodes.append(node)
            else:
                text_nodes.append(node)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
        ) as event:
            query_str = query.query_str
            if query.chat_messages_str:
                query_str = query.chat_messages_str + "\nassistant: "

            if query.no_retrieval:
                response_str = self.get_llm_only_response(
                    query_str=query_str,
                    streaming=query.stream,
                    system_role_str=system_role_str or self._system_role_template,
                    prompt_template_str=prompt_template_str
                    or self._custom_prompt_template,
                    **response_kwargs,
                )
            else:
                response_str = self.get_response(
                    query_str=query_str,
                    text_chunks=[
                        n.node.get_content(metadata_mode=MetadataMode.LLM)
                        for n in text_nodes
                    ],
                    image_url_list=[n.node.image_url for n in image_nodes],
                    streaming=query.stream,
                    citation=query.citation,
                    system_role_str=system_role_str or self._system_role_template,
                    prompt_template_str=prompt_template_str
                    or self._custom_prompt_template,
                    **response_kwargs,
                )

            additional_source_nodes = additional_source_nodes or []
            source_nodes = list(nodes) + list(additional_source_nodes)

            response = self._prepare_response_output(response_str, source_nodes)

            event.on_end(payload={EventPayload.RESPONSE: response})

        dispatcher.event(
            SynthesizeEndEvent(
                query=query,
                response=response,
            )
        )
        return response

    @dispatcher.span
    async def asynthesize(
        self,
        query: PaiQueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        system_role_str: str = None,
        prompt_template_str: str = None,
        **response_kwargs: Any,
    ) -> ChatResponseWrapper:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )

        if isinstance(query, str):
            query = QueryBundle(query_str=query)

        text_nodes, image_nodes = [], []
        for node in nodes:
            if isinstance(node.node, ImageNode):
                image_nodes.append(node)
            else:
                text_nodes.append(node)

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
        ) as event:
            query_str = query.query_str
            if query.chat_messages_str:
                query_str = query.chat_messages_str + "\nassistant: "
            if query.no_retrieval:
                response = await self.aget_llm_only_response(
                    query_str=query_str,
                    streaming=query.stream,
                    system_role_str=system_role_str or self._system_role_template,
                    prompt_template_str=prompt_template_str
                    or self._custom_prompt_template,
                    **response_kwargs,
                )
            else:
                response = await self.aget_response(
                    query_str=query_str,
                    text_chunks=[
                        n.node.get_content(metadata_mode=MetadataMode.LLM)
                        for n in text_nodes
                    ],
                    image_url_list=[n.node.image_url for n in image_nodes],
                    streaming=query.stream,
                    citation=query.citation,
                    system_role_str=system_role_str or self._system_role_template,
                    prompt_template_str=prompt_template_str
                    or self._custom_prompt_template,
                    **response_kwargs,
                )

            additional_source_nodes = additional_source_nodes or []
            source_nodes = list(nodes) + list(additional_source_nodes)

            event.on_end(payload={EventPayload.RESPONSE: response})

        return ChatResponseWrapper(response=response, source_nodes=source_nodes)

    def _get_multi_modal_response(
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
            completion_response_gen = self._multimodal_llm.stream_complete(
                prompt=fmt_prompt,
                image_documents=image_documents,
                **response_kwargs,
            )
            stream_tokens = stream_completion_response_to_tokens(
                completion_response_gen
            )
            return cast(Generator, stream_tokens)
        else:
            llm_response = self._multimodal_llm.complete(
                prompt=fmt_prompt,
                image_documents=image_documents,
                **response_kwargs,
            )
            response = llm_response.text or DEFAULT_EMPTY_RESPONSE_GEN
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

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
        citation: bool = False,
        system_role_str: str = None,
        prompt_template_str: str = None,
        **response_kwargs: Any,
    ) -> Union[ChatResponse, ChatResponseAsyncGen]:
        if image_url_list and len(image_url_list) > 0:
            assert (
                self._multimodal_llm is not None
            ), "Multi-modal LLM must be provided to understand image documents."

            return await self._aget_multi_modal_response(
                query_str=query_str,
                text_chunks=text_chunks,
                image_url_list=image_url_list,
                streaming=streaming,
                citation=citation,
                **response_kwargs,
            )

        logger.info(f"Synthsize using LLM with no image inputs. citation: {citation}")
        if not citation:
            prompt_template = (
                PromptTemplate(
                    template="{}\n{}\n{}\n{}".format(
                        system_role_str,
                        CURRENT_TIME_PROMPT.format(
                            current_datetime=datetime.now().strftime(
                                "%Y年%m月%d日 %H:%M:%S"
                            )
                        ),
                        prompt_template_str,
                        DEFAULT_CONTEXT_ANSWER_TEMPLATE,
                    )
                )
                or self._text_qa_template
            )
        else:
            prompt_template = (
                PromptTemplate(
                    template="{}\n{}\n{}\n{}\n{}".format(
                        system_role_str,
                        CURRENT_TIME_PROMPT.format(
                            current_datetime=datetime.now().strftime(
                                "%Y年%m月%d日 %H:%M:%S"
                            )
                        ),
                        prompt_template_str,
                        DEFAULT_CUSTOM_CITATION_PROMPR_TEMPLATE,
                        DEFAULT_CONTEXT_ANSWER_TEMPLATE,
                    )
                )
                or self._citation_text_qa_template
            )

        text_qa_template = prompt_template.partial_format(query_str=query_str)

        context_str = "\n".join(
            [f"材料 {i+1}:\n{text}\n" for i, text in enumerate(text_chunks)]
        )

        response: RESPONSE_TEXT_TYPE
        logger.info(
            f"Synthsize using LLM with contexts. \n Prompt: {text_qa_template} \n Query: {query_str}"
        )
        messages = self._llm._get_messages(
            text_qa_template,
            context_str=context_str,
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

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
        citation: bool = False,
        system_role_str: str = None,
        prompt_template_str: str = None,
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        if image_url_list and len(image_url_list) > 0:
            assert (
                self._multimodal_llm is not None
            ), "Multi-modal LLM must be provided to understand image documents."
            return self._get_multi_modal_response(
                query_str=query_str,
                text_chunks=text_chunks,
                image_url_list=image_url_list,
                streaming=streaming,
                **kwargs,
            )

        if not citation:
            prompt_template = (
                PromptTemplate(
                    template="{}\n{}\n{}\n{}".format(
                        system_role_str,
                        CURRENT_TIME_PROMPT.format(
                            current_datetime=datetime.now().strftime(
                                "%Y年%m月%d日 %H:%M:%S"
                            )
                        ),
                        prompt_template_str,
                        DEFAULT_CONTEXT_ANSWER_TEMPLATE,
                    )
                )
                or self._text_qa_template
            )
        else:
            prompt_template = (
                PromptTemplate(
                    template="{}\n{}\n{}\n{}\n{}".format(
                        system_role_str,
                        CURRENT_TIME_PROMPT.format(
                            current_datetime=datetime.now().strftime(
                                "%Y年%m月%d日 %H:%M:%S"
                            )
                        ),
                        prompt_template_str,
                        DEFAULT_CUSTOM_CITATION_PROMPR_TEMPLATE,
                        DEFAULT_CONTEXT_ANSWER_TEMPLATE,
                    )
                )
                or self._citation_text_qa_template
            )

        text_qa_template = prompt_template.partial_format(query_str=query_str)
        context_str = "\n".join(
            [f"材料 {i+1}:\n{text}\n" for i, text in enumerate(text_chunks)]
        )

        response: RESPONSE_TEXT_TYPE
        logger.info(f"Synthsize using LLM with contexts. \n Prompt: {text_qa_template}")
        if not streaming:
            response = self._llm.predict(
                text_qa_template,
                context_str=context_str,
                **kwargs,
            )
        else:
            response = self._llm.stream(
                text_qa_template,
                context_str=context_str,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or DEFAULT_EMPTY_RESPONSE_GEN
        else:
            response = cast(Generator, response)

        return response

    async def aget_llm_only_response(
        self,
        query_str: str,
        streaming: bool = False,
        system_role_str: str = None,
        prompt_template_str: str = None,
        **kwargs: Any,
    ) -> Union[ChatResponse, ChatResponseAsyncGen]:
        response: RESPONSE_TEXT_TYPE
        _llm_only_template = PromptTemplate(
            template="{}\n{}\n{}\n{}".format(
                system_role_str,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
                ),
                prompt_template_str,
                DEFAULT_ANSWER_TEMPLATE,
            )
        )
        logger.info(
            f"Synthsize using LLM only. \n Prompt: {_llm_only_template}. \n Query: {query_str}"
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

    def get_llm_only_response(
        self,
        query_str: str,
        streaming: bool = False,
        system_role_str: str = None,
        prompt_template_str: str = None,
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        response: RESPONSE_TEXT_TYPE

        _llm_only_template = PromptTemplate(
            template="{}\n{}\n{}\n{}".format(
                system_role_str,
                CURRENT_TIME_PROMPT.format(
                    current_datetime=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
                ),
                prompt_template_str,
                DEFAULT_ANSWER_TEMPLATE,
            )
        )
        logger.info(
            f"Synthsize using LLM only. \n Prompt: {_llm_only_template}. \n Query: {query_str}"
        )
        if not streaming:
            response = self._llm.predict(
                _llm_only_template,
                query_str=query_str,
                **kwargs,
            )
        else:
            response = self._llm.stream(
                _llm_only_template,
                query_str=query_str,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or DEFAULT_EMPTY_RESPONSE_GEN
        else:
            response = cast(Generator, response)

        return response
