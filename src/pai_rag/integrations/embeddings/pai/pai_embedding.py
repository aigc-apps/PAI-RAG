from typing import Any, Callable, List, Optional
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, ImageNode

import logging

from pai_rag.integrations.embeddings.pai.embedding_utils import create_embedding
from pai_rag.integrations.embeddings.pai.pai_embedding_config import (
    PaiBaseEmbeddingConfig,
)

logger = logging.getLogger(__name__)


class PaiEmbedding(BaseEmbedding):
    """PAI embedding model"""

    _embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)
    _embed_model: Any = PrivateAttr()
    _config: Any = PrivateAttr()

    def __init__(
        self,
        embed_config: PaiBaseEmbeddingConfig,
    ):
        self._config = embed_config
        self._embed_model = create_embedding(embed_config)

        super().__init__(
            model_name=self._embed_model.model_name,
            embed_batch_size=self._embed_model.embed_batch_size,
            callback_manager=self._embed_model.callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "PaiEmbedding"

    async def _aget_query_embedding(self, query: str) -> Embedding:
        raise NotImplementedError

    def _get_query_embedding(self, text: str) -> Embedding:
        raise NotImplementedError

    def _get_text_embedding(self, text: str) -> Embedding:
        raise NotImplementedError

    async def aget_query_embedding(self, query: str) -> Embedding:
        return await self._embed_model.aget_query_embedding(query)

    async def aget_text_embedding(self, text: str) -> Embedding:
        return await self._embed_model.aget_text_embedding(text)

    async def aget_agg_embedding_from_queries(
        self,
        queries: List[str],
        agg_fn: Optional[Callable[..., Embedding]] = None,
    ) -> Embedding:
        return await self._embed_model.aget_agg_embedding_from_queries(queries, agg_fn)

    async def aget_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> List[List[float]]:
        return await self._embed_model.aget_text_embedding_batch(texts, show_progress)

    def get_query_embedding(self, query: str) -> List[float]:
        return self._embed_model.get_query_embedding(query)

    def get_text_embedding(self, text: str) -> List[float]:
        return self._embed_model.get_text_embedding(text)

    def get_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False, **kwargs: Any
    ) -> List[List[float]]:
        return self._embed_model.get_text_embedding_batch(
            texts, show_progress, **kwargs
        )

    def get_agg_embedding_from_queries(
        self, queries: List[str], agg_fn: Callable[..., List[float]] | None = None
    ) -> List[float]:
        return self._embed_model.get_agg_embedding_from_queries(queries, agg_fn)

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        text_node_ids, texts = [], []
        for i, node in enumerate(nodes):
            if not isinstance(node, ImageNode):
                text_node_ids.append(i)
                texts.append(node.get_content(metadata_mode=MetadataMode.EMBED))

        embeddings = self.get_text_embedding_batch(texts=texts, **kwargs)
        for i, embedding in zip(text_node_ids, embeddings):
            nodes[i].embedding = embedding

        return nodes

    async def acall(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        text_node_ids, texts = [], []
        for i, node in enumerate(nodes):
            if not isinstance(node, ImageNode):
                text_node_ids.append(i)
                texts.append(node.get_content(metadata_mode=MetadataMode.EMBED))

        embeddings = await self.aget_text_embedding_batch(texts=texts, **kwargs)
        for i, embedding in zip(text_node_ids, embeddings):
            nodes[i].embedding = embedding

        return nodes

    def __hash__(self):
        return hash(self._config)
