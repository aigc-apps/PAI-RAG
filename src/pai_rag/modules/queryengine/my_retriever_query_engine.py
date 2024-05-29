from typing import List

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine

import llama_index.core.instrumentation as instrument
import logging

dispatcher = instrument.get_dispatcher(__name__)
logger = logging.getLogger(__name__)


class MyRetrieverQueryEngine(RetrieverQueryEngine):
    """Modified retriever query engine, modify retrieve/aretrieve func to identify routered retriever type, 0 indicates summary_retriever, 1 indicates fusion_retriever

    Args:
        retriever (BaseRetriever): A retriever object.
        response_synthesizer (Optional[BaseSynthesizer]): A BaseSynthesizer
            object.
        callback_manager (Optional[CallbackManager]): A callback manager.
    """

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        try:
            result = self._retriever._selector.select(
                self._retriever._metadatas, query_bundle
            )
            if result.ind == 0:
                return nodes
            else:
                return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)
        except Exception as ex:
            logger.warn(f"{ex}")
            return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)
        try:
            result = await self._retriever._selector.aselect(
                self._retriever._metadatas, query_bundle
            )
            if result.ind == 0:
                return nodes
            else:
                return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)
        except Exception as ex:
            logger.warn(f"{ex}")
            return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    @dispatcher.span
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = self.retrieve(query_bundle)
            response = self._response_synthesizer.synthesize(
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

            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
