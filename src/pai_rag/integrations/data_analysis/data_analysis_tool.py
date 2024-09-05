import logging
from typing import Optional, List


from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.settings import Settings
import llama_index.core.instrumentation as instrument


from pai_rag.integrations.data_analysis.nl2pandas_retriever import PandasQueryRetriever
from pai_rag.integrations.data_analysis.data_analysis_synthesizer import (
    DataAnalysisSynthesizer,
)

logger = logging.getLogger(__name__)

dispatcher = instrument.get_dispatcher(__name__)


class DataAnalysisTool(BaseQueryEngine):
    """
    Used for db or excel/csv file Data Analysis
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        analysis_retriever: BaseRetriever = PandasQueryRetriever,
        analysis_synthesizer: BaseSynthesizer = DataAnalysisSynthesizer,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Initialize params."""
        self._llm = llm or Settings.llm
        self._retriever = analysis_retriever
        self._synthesizer = analysis_synthesizer
        super().__init__(callback_manager=callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        return nodes

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)
        return nodes

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
    ) -> RESPONSE_TYPE:
        return self._synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
    ) -> RESPONSE_TYPE:
        return await self._synthesizer.asynthesize(
            query=query_bundle,
            nodes=nodes,
        )

    @dispatcher.span
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = self.retrieve(query_bundle)
            response = self._synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )
            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    @dispatcher.span
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = await self.aretrieve(query_bundle)
            response = await self._synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def astream_query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        streaming = self._synthesizer._streaming
        self._synthesizer._streaming = True

        nodes = await self.aretrieve(query_bundle)

        stream_response = await self._synthesizer.asynthesize(
            query=query_bundle, nodes=nodes
        )
        self._synthesizer._streaming = streaming

        return stream_response
