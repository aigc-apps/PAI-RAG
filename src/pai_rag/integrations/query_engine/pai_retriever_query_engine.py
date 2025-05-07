from typing import List, Optional, Sequence, Union
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import NodeWithScore, QueryBundle, ImageNode, QueryType
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.callbacks.base import CallbackManager
import llama_index.core.instrumentation as instrument
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.instrumentation.events.query import (
    QueryStartEvent,
)
from llama_index.core.base.llms.types import (
    ChatResponse,
    ChatResponseAsyncGen,
)
from pai_rag.app.api.models import ChatResponseWrapper, PaiQueryBundle

dispatcher = instrument.get_dispatcher(__name__)


class PaiQueryEndEvent(BaseEvent):
    """QueryEndEvent.

    Args:
        query (QueryType): Query as a string or query bundle.
        response (RESPONSE_TYPE): Response.
    """

    query: QueryType
    response: ChatResponseWrapper

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "PaiQueryEndEvent"


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
        super().__init__(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
            callback_manager=callback_manager,
        )

        self._query_transform = query_transform
        self._transform_metadata = transform_metadata

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

    def _query(
        self,
        query_bundle: PaiQueryBundle,
        system_role_str: str = None,
        prompt_template_str: str = None,
    ) -> RESPONSE_TYPE:
        """Answer a query."""
        nodes = self.retrieve(query_bundle)
        response = self._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
            system_role_str=system_role_str,
            prompt_template_str=prompt_template_str,
        )

        return response

    async def _aquery(
        self,
        query_bundle: PaiQueryBundle,
    ) -> RESPONSE_TYPE:
        """Answer a query."""
        nodes = await self.aretrieve(query_bundle)
        response = await self._response_synthesizer.asynthesize(
            query=query_bundle,
            nodes=nodes,
            system_role_str=query_bundle.system_role,
            prompt_template_str=" " if query_bundle.system_role else None,
            **query_bundle.llm_kwargs
        )

        return response

    def query(
        self,
        str_or_query_bundle: QueryType,
        system_role_str: str = None,
        prompt_template_str: str = None,
    ) -> RESPONSE_TYPE:
        raise NotImplementedError

    async def aquery(self, query_bundle: QueryType) -> RESPONSE_TYPE:
        dispatcher.event(QueryStartEvent(query=query_bundle))
        if isinstance(query_bundle, str):
            query_bundle = PaiQueryBundle(query_bundle)
        query_result = await self._aquery(query_bundle)
        print("+++ query engine returns")

        if not query_bundle.stream:
            dispatcher.event(
                PaiQueryEndEvent(query=query_bundle, response=query_result)
            )
        return query_result

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        system_role_str: str = None,
        prompt_template_str: str = None,
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        return self._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
            system_role_str=system_role_str,
            prompt_template_str=prompt_template_str,
            additional_source_nodes=additional_source_nodes,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        system_role_str: str = None,
        prompt_template_str: str = None,
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> Union[ChatResponse, ChatResponseAsyncGen]:
        return await self._response_synthesizer.asynthesize(
            query=query_bundle,
            nodes=nodes,
            system_role_str=system_role_str,
            prompt_template_str=prompt_template_str,
            additional_source_nodes=additional_source_nodes,
        )
