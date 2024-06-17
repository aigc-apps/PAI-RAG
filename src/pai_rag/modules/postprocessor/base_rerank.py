from typing import Any, List, Optional

from llama_index.core.base.query_pipeline.query import (
    QueryComponent,
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import (
    BaseNodePostprocessor,
    PostprocessorComponent,
)


# Add async interface only.
# No implementation provided.
class CustomNodePostprocessor(BaseNodePostprocessor):
    # implement class_name so users don't have to worry about it when extending
    @classmethod
    def class_name(cls) -> str:
        return "CustomNodePostprocessor"

    async def postprocess_nodes_async(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        if query_str is not None and query_bundle is not None:
            raise ValueError("Cannot specify both query_str and query_bundle")
        elif query_str is not None:
            query_bundle = QueryBundle(query_str)
        else:
            pass
        return await self._postprocess_nodes_async(nodes, query_bundle)

    async def _postprocess_nodes_async(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        raise NotImplementedError

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        raise NotImplementedError

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """As query component."""
        return PostprocessorComponent(postprocessor=self)
