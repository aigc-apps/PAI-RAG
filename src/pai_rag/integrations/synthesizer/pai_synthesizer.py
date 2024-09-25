from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Sequence, AsyncGenerator, cast

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_TEXT_QA_PROMPT_SEL,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
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
    Response,
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.instrumentation.events.synthesis import (
    SynthesizeStartEvent,
    SynthesizeEndEvent,
)
from llama_index.core.llms.llm import (
    stream_completion_response_to_tokens,
    astream_completion_response_to_tokens,
)
from llama_index.core.prompts import PromptTemplate
import logging

dispatcher = instrument.get_dispatcher(__name__)

logger = logging.getLogger(__name__)

DEFAULT_LLM_CHAT_TMPL = (
    "You are a helpful assistant."
    "Please answer the following question. \n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL = (
    "结合上面给出的图片和下面给出的参考材料来回答用户的问题。材料中包含一组图片链接，分别对应到前面给出的图片的地址。\n\n"
    "材料:"
    "---------------------\n\n"
    "{context_str}\n"
    "---------------------\n\n"
    "请根据给定的材料回答给出的问题，回答中需要有文字描述和图片。如果材料中没有找到答案，就说没有找到相关的信息，不要编造答案。\n\n"
    "如果上面有图片对你生成答案有帮助，请找到图片链接并用markdown格式给出，如![](image_url)。\n\n"
    "---------------------\n\n"
    "问题: {query_str}\n请返回文字和展示图片，不需要标明图片顺序"
    "答案: "
)
QueryTextType = QueryType


def empty_response_generator() -> Generator[str, None, None]:
    yield "Empty Response"


async def empty_response_agenerator() -> AsyncGenerator[str, None]:
    yield "Empty Response"


@dataclass
class PaiQueryBundle(QueryBundle):
    stream: bool = False
    no_retrieval: bool = False


"""
PaiSynthesizer:
Supports multi-modal inputs synthesizer.
Will use Multi-modal LLM for inputs with images and LLM for pure text inputs.
"""


class PaiSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        llm: Optional[LLMPredictorType] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        multimodal_llm: Optional[MultiModalLLM] = None,
        multimodal_qa_template: Optional[BasePromptTemplate] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            streaming=streaming,
        )
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
        self._multimodal_qa_template = multimodal_qa_template or PromptTemplate(
            template=DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL
        )
        self._llm_only_template = PromptTemplate(template=DEFAULT_LLM_CHAT_TMPL)
        self._multimodal_llm = multimodal_llm

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"text_qa_template": self._text_qa_template}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_qa_template" in prompts:
            self._text_qa_template = prompts["text_qa_template"]

    @dispatcher.span
    def synthesize(
        self,
        query: PaiQueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )

        if not query.no_retrieval and len(nodes) == 0:
            if query.stream:
                empty_response = StreamingResponse(
                    response_gen=empty_response_generator()
                )
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response
            else:
                empty_response = Response("Empty Response")
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response

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
            if query.no_retrieval:
                response_str = self.get_llm_only_response(
                    query_str=query.query_str,
                    streaming=query.stream,
                    **response_kwargs,
                )
            else:
                response_str = self.get_response(
                    query_str=query.query_str,
                    text_chunks=[
                        n.node.get_content(metadata_mode=MetadataMode.LLM)
                        for n in text_nodes
                    ],
                    image_url_list=[n.node.image_url for n in image_nodes],
                    streaming=query.stream,
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
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )
        if not query.no_retrieval and len(nodes) == 0:
            if query.stream:
                empty_response = AsyncStreamingResponse(
                    response_gen=empty_response_agenerator()
                )
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response
            else:
                empty_response = Response("Empty Response")
                dispatcher.event(
                    SynthesizeEndEvent(
                        query=query,
                        response=empty_response,
                    )
                )
                return empty_response

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
            if query.no_retrieval:
                response_str = await self.aget_llm_only_response(
                    query_str=query.query_str,
                    streaming=query.stream,
                    **response_kwargs,
                )
            else:
                response_str = await self.aget_response(
                    query_str=query.query_str,
                    text_chunks=[
                        n.node.get_content(metadata_mode=MetadataMode.LLM)
                        for n in text_nodes
                    ],
                    image_url_list=[n.node.image_url for n in image_nodes],
                    streaming=query.stream,
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

    def _get_multi_modal_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        image_documents = load_image_urls(image_url_list)
        image_context_str = "\n\n".join(image_url_list)
        text_context_str = "\n\n".join(text_chunks)
        context_str = f"{text_context_str}\n\n图片链接列表: \n\n{image_context_str}\n\n"
        fmt_prompt = self._multimodal_qa_template.format(
            context_str=context_str, query_str=query_str
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
            response = llm_response.text or "Empty Response"
            return response

    async def _aget_multi_modal_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        image_documents = load_image_urls(image_url_list)

        image_context_str = "\n\n".join(image_url_list)
        text_context_str = "\n\n".join(text_chunks)
        context_str = f"{text_context_str}\n\n图片链接列表: \n\n{image_context_str}\n\n"
        fmt_prompt = self._multimodal_qa_template.format(
            context_str=context_str, query_str=query_str
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
            response = llm_response.text or "Empty Response"
            return response

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        if image_url_list and len(image_url_list) > 0:
            assert (
                self._multimodal_llm is not None
            ), "Multi-modal LLM must be provided to understand image documents."

            logger.info(
                f"Synthsize using Multi-modal LLM with images {image_url_list}."
            )
            return await self._aget_multi_modal_response(
                query_str=query_str,
                text_chunks=text_chunks,
                image_url_list=image_url_list,
                streaming=streaming,
                **response_kwargs,
            )

        logger.info("Synthsize using LLM with no image inputs.")
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        single_text_chunk = "\n".join(text_chunks)
        truncated_chunks = self._prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=[single_text_chunk],
        )

        response: RESPONSE_TEXT_TYPE
        if not streaming:
            response = await self._llm.apredict(
                text_qa_template,
                context_str=truncated_chunks,
                **response_kwargs,
            )
        else:
            # customized modify [will be removed]
            response = await self._llm.astream(
                text_qa_template,
                context_str=truncated_chunks,
                **response_kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        image_url_list: Sequence[str] = None,
        streaming: bool = False,
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

        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        single_text_chunk = "\n".join(text_chunks)
        truncated_chunks = self._prompt_helper.truncate(
            prompt=text_qa_template,
            text_chunks=[single_text_chunk],
        )

        response: RESPONSE_TEXT_TYPE
        if not streaming:
            response = self._llm.predict(
                text_qa_template,
                context_str=truncated_chunks,
                **kwargs,
            )
        else:
            response = self._llm.stream(
                text_qa_template,
                context_str=truncated_chunks,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    async def aget_llm_only_response(
        self,
        query_str: str,
        streaming: bool = False,
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        response: RESPONSE_TEXT_TYPE
        if not streaming:
            response = await self._llm.apredict(
                self._llm_only_template,
                query_str=query_str,
                **kwargs,
            )
        else:
            response = await self._llm.astream(
                self._llm_only_template,
                query_str=query_str,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    def get_llm_only_response(
        self,
        query_str: str,
        streaming: bool = False,
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        response: RESPONSE_TEXT_TYPE
        if not streaming:
            response = self._llm.predict(
                self._llm_only_template,
                query_str=query_str,
                **kwargs,
            )
        else:
            response = self._llm.stream(
                self._llm_only_template,
                query_str=query_str,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response
