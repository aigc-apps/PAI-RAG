from dataclasses import dataclass
from typing import List, Optional
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import NodeWithScore, QueryBundle, ImageNode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.callbacks.base import CallbackManager
import llama_index.core.instrumentation as instrument
from llama_index.core.response_synthesizers import BaseSynthesizer
import logging

dispatcher = instrument.get_dispatcher(__name__)
logger = logging.getLogger(__name__)


@dataclass
class PaiQueryBundle(QueryBundle):
    stream: bool = False


class PaiRetrieverQueryEngine(RetrieverQueryEngine):
    """Retriever query engine.

        pplies a query transform to a query bundle before passing
        it to a query engine.

    Args:
        query_transform (BaseQueryTransform): A query transform object.
        transform_metadata (Optional[dict]): metadata to pass to the
            query transform.
        callback_manager (Optional[CallbackManager]): A callback manager.

    """

    def __init__(
        self,
        retriever: BaseRetriever,
        query_transform: Optional[BaseQueryTransform] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        transform_metadata: Optional[dict] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._query_transform = query_transform
        self._transform_metadata = transform_metadata
        super().__init__(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
            callback_manager=callback_manager,
        )

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self._query_transform:
            query_bundle = self._query_transform.run(
                query_bundle, metadata=self._transform_metadata
            )
        nodes = self._retriever.retrieve(query_bundle)
        text_nodes, image_nodes = [], []
        for node in nodes:
            if isinstance(node.node, ImageNode):
                image_nodes.append(node)
            else:
                text_nodes.append(node)

        text_nodes = self._apply_node_postprocessors(
            text_nodes, query_bundle=query_bundle
        )
        return [n for n in text_nodes] + image_nodes

    # 支持异步
    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self._query_transform:
            query_bundle = await self._query_transform.arun(
                query_bundle, metadata=self._transform_metadata
            )
        nodes = await self._retriever.aretrieve(query_bundle)
        text_nodes, image_nodes = [], []
        for node in nodes:
            if isinstance(node.node, ImageNode):
                image_nodes.append(node)
            else:
                text_nodes.append(node)

        for node_postprocessor in self._node_postprocessors:
            text_nodes = node_postprocessor.postprocess_nodes(
                text_nodes,
                query_bundle=query_bundle,
            )

        return [n for n in text_nodes] + image_nodes

    @dispatcher.span
    def _query(self, query_bundle: PaiQueryBundle) -> RESPONSE_TYPE:
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
    async def _aquery(self, query_bundle: PaiQueryBundle) -> RESPONSE_TYPE:
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
