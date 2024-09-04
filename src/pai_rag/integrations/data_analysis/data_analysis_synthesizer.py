import logging
from typing import Any, List, Generator, Optional, Sequence, cast, AsyncGenerator

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.schema import NodeWithScore, QueryType, QueryBundle
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.service_context import ServiceContext
from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
from llama_index.core.types import RESPONSE_TEXT_TYPE
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
from llama_index.core.callbacks.schema import CBEventType, EventPayload
import llama_index.core.instrumentation as instrument

logger = logging.getLogger(__name__)

dispatcher = instrument.get_dispatcher(__name__)


def empty_response_generator() -> Generator[str, None, None]:
    yield "Empty Response"


async def empty_response_agenerator() -> AsyncGenerator[str, None]:
    yield "Empty Response"


DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL = (
    "Given an input question, synthesize a response in Chinese from the query results.\n"
    "Query: {query_str}\n\n"
    "SQL or Python Code Instructions (optional):\n{query_code_instruction}\n\n"
    "Code Query Output: {query_output}\n\n"
    "Response: "
)

DEFAULT_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_TMPL,
)


class DataAnalysisSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        llm: Optional[LLMPredictorType] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        response_synthesis_prompt: Optional[BasePromptTemplate] = None,
        streaming: bool = False,
        # deprecated
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        logger.info("DataAnalysisSynthesizer initialized")
        if service_context is not None:
            prompt_helper = service_context.prompt_helper

        self._llm = llm or Settings.llm
        self._response_synthesis_prompt = (
            response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        )

        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            service_context=service_context,
            streaming=streaming,
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"response_synthesis_prompt": self._response_synthesis_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "response_synthesis_prompt" in prompts:
            self._response_synthesis_prompt = prompts["response_synthesis_prompt"]

    async def aget_response(
        self,
        query_str: str,
        retrieved_nodes: List[NodeWithScore],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        query_df_output = [n.node.get_content() for n in retrieved_nodes]

        partial_prompt_tmpl = self._response_synthesis_prompt.partial_format(
            query_str=query_str,
            query_code_instruction=[
                n.node.metadata["query_code_instruction"] for n in retrieved_nodes
            ],
        )
        truncated_df_output = self._prompt_helper.truncate(
            prompt=partial_prompt_tmpl,
            text_chunks=["\n".join(query_df_output)],
        )
        logger.info(f"truncated_df_output: {str(truncated_df_output)}")

        response: RESPONSE_TEXT_TYPE
        if not self._streaming:
            response = await self._llm.apredict(
                self._response_synthesis_prompt,
                query_str=query_str,
                query_code_instruction=[
                    n.node.metadata["query_code_instruction"] for n in retrieved_nodes
                ],  # sql or pandas query
                query_output=truncated_df_output,  # query output
                **response_kwargs,
            )
        else:
            response = await self._llm.astream(
                self._response_synthesis_prompt,
                query_str=query_str,
                query_code_instruction=[
                    n.node.metadata["query_code_instruction"] for n in retrieved_nodes
                ],
                query_output=truncated_df_output,
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
        retrieved_nodes: List[NodeWithScore],
        **kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        query_df_output = [n.node.get_content() for n in retrieved_nodes]

        partial_prompt_tmpl = self._response_synthesis_prompt.partial_format(
            query_str=query_str,
            query_code_instruction=[
                n.node.metadata["query_code_instruction"] for n in retrieved_nodes
            ],
        )
        truncated_df_output = self._prompt_helper.truncate(
            prompt=partial_prompt_tmpl,
            text_chunks=["\n".join(query_df_output)],
        )
        logger.info(f"truncated_df_output: {truncated_df_output}")

        response: RESPONSE_TEXT_TYPE
        if not self._streaming:
            response = self._llm.predict(
                self._response_synthesis_prompt,
                query_str=query_str,
                query_code_instruction=[
                    n.node.metadata["query_code_instruction"] for n in retrieved_nodes
                ],  # sql or pandas query
                query_output=truncated_df_output,  # query output
                **kwargs,
            )
        else:
            response = self._llm.stream(
                self._response_synthesis_prompt,
                query_str=query_str,
                query_code_instruction=[
                    n.node.metadata["query_code_instruction"] for n in retrieved_nodes
                ],
                query_output=truncated_df_output,
                **kwargs,
            )

        if isinstance(response, str):
            response = response or "Empty Response"
        else:
            response = cast(Generator, response)

        return response

    @dispatcher.span
    def synthesize(
        self,
        query: QueryType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )

        if len(nodes) == 0:
            if self._streaming:
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

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
        ) as event:
            response_str = self.get_response(
                query_str=query.query_str,
                retrieved_nodes=nodes,
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
        query: QueryType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        dispatcher.event(
            SynthesizeStartEvent(
                query=query,
            )
        )
        if len(nodes) == 0:
            if self._streaming:
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

        with self._callback_manager.event(
            CBEventType.SYNTHESIZE,
            payload={EventPayload.QUERY_STR: query.query_str},
        ) as event:
            response_str = await self.aget_response(
                query_str=query.query_str,
                retrieved_nodes=nodes,
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
