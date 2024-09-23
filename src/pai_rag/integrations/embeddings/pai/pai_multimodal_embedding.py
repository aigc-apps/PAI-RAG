from io import BytesIO
from typing import Any, Callable, List, Optional
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import ImageType, BaseNode, ImageNode
import logging
from pai_rag.integrations.embeddings.pai.pai_embedding_config import (
    PaiBaseEmbeddingConfig,
)
from pai_rag.integrations.embeddings.pai.embedding_utils import create_embedding

logger = logging.getLogger(__name__)


class PaiMultiModalEmbedding(MultiModalEmbedding):
    """PAI multimodal embedding model"""

    _embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)
    _embed_model: Any = PrivateAttr()
    _config: Any = PrivateAttr()

    def __init__(
        self,
        multi_modal_embed_config: PaiBaseEmbeddingConfig,
    ):
        self._config = multi_modal_embed_config
        self._embed_model = create_embedding(multi_modal_embed_config)

        super().__init__(
            model_name=self._embed_model.model_name,
            embed_batch_size=self._embed_model.embed_batch_size,
            callback_manager=self._embed_model.callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "PaiMultiModalEmbedding"

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

    # Multi-Modal interfaces
    def get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """
        Embed the input image.
        """
        return self._embed_model.get_image_embedding(img_file_path=img_file_path)

    async def aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """Get image embedding."""
        return await self._embed_model.aget_image_embedding(img_file_path=img_file_path)

    def get_image_embedding_batch(
        self, img_file_paths: List[ImageType], show_progress: bool = False
    ) -> List[Embedding]:
        """Get a list of image embeddings, with batching."""
        return self._embed_model.get_image_embedding_batch(
            img_file_paths=img_file_paths
        )

    async def aget_image_embedding_batch(
        self, img_file_paths: List[ImageType], show_progress: bool = False
    ) -> List[Embedding]:
        """Get a list of image embeddings, with batching."""
        return await self._embed_model.aget_image_embedding_batch(
            img_file_paths=img_file_paths
        )

    def _get_image_embedding(self, img_file_path: str | BytesIO) -> List[float]:
        return self._embed_model._get_image_embedding(img_file_path)

    async def _aget_image_embedding(self, img_file_path: str | BytesIO) -> List[float]:
        return await self._embed_model._aget_image_embedding(img_file_path)

    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        image_node_ids, images = [], []
        for i, node in enumerate(nodes):
            if isinstance(node, ImageNode):
                image_node_ids.append(i)
                images.append(node.resolve_image())

        embeddings = self.get_image_embedding_batch(img_file_paths=images, **kwargs)
        for i, embedding in zip(image_node_ids, embeddings):
            nodes[i].embedding = embedding

        return nodes

    async def acall(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        image_node_ids, images = [], []
        for i, node in enumerate(nodes):
            if isinstance(node, ImageNode):
                image_node_ids.append(i)
                images.append(node.resolve_image())

        embeddings = await self.aget_image_embedding_batch(
            img_file_paths=images, **kwargs
        )
        for i, embedding in zip(image_node_ids, embeddings):
            nodes[i].embedding = embedding

        return nodes

    def __hash__(self):
        return hash(self._config)
