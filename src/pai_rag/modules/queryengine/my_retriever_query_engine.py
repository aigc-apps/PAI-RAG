from typing import List

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine

import llama_index.core.instrumentation as instrument
import logging
import datetime

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
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    # 支持异步
    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        print(f"{datetime.datetime.now()} start retrieving ====")
        nodes = await self._retriever.aretrieve(query_bundle)
        print(f"{datetime.datetime.now()} finish retrieving ====")

        for node_postprocessor in self._node_postprocessors:
            nodes = await node_postprocessor.postprocess_nodes_async(
                # nodes = node_postprocessor.postprocess_nodes(
                nodes,
                query_bundle=query_bundle,
            )
        return nodes

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
            start = datetime.datetime.now()
            print(f"{start} start retriever ====")

            nodes = await self.aretrieve(query_bundle)
            print(f"Finished retriever ====, costed: {datetime.datetime.now()-start}")

            start = datetime.datetime.now()
            print(f"{start} start synthesize ====")

            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            print(f"Finished synthesize ====, costed: {datetime.datetime.now()-start}")

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
