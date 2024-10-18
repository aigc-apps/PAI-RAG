"""Base vector store index query."""

import asyncio
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_multi_modal_retriever import (
    MultiModalRetriever,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.indices.utils import log_vector_store_query_result
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
from llama_index.core.schema import (
    NodeWithScore,
    QueryBundle,
    QueryType,
    ImageNode,
)
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from pai_rag.integrations.index.pai.local.local_bm25_index import LocalBm25IndexStore
import logging
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)
logger = logging.getLogger(__name__)

DEFAULT_IMAGE_STORE = "image"


class PaiMultiModalVectorIndexRetriever(MultiModalRetriever):
    """Multi Modal Vector index retriever.

    Args:
        index (MultiModalVectorStoreIndex): Multi Modal vector store index for images and texts.
        similarity_top_k (int): number of top k results to return.
        vector_store_query_mode (str): vector store query mode
            See reference for VectorStoreQueryMode for full list of supported modes.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        alpha (float): weight for sparse/dense retrieval, only used for
            hybrid query mode.
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        vector_store_kwargs (dict): Additional vector store specific kwargs to pass
            through to the vector store at query time.

    """

    def __init__(
        self,
        index: Any,
        enable_multimodal: bool = True,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        image_similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        filters: Optional[MetadataFilters] = None,
        alpha: Optional[float] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        sparse_top_k: Optional[int] = None,
        search_image=False,
        supports_hybrid_search=True,
        local_bm25_index: LocalBm25IndexStore = None,
        hybrid_fusion_weights: List[float] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._vector_store = self._index.vector_store
        # separate image vector store for image retrieval
        self._enable_multimodal = enable_multimodal

        self._image_vector_store = None

        if self._enable_multimodal:
            self._image_vector_store = self._index.image_vector_store

            assert isinstance(self._index.image_embed_model, BaseEmbedding)
            self._image_embed_model = index._image_embed_model
        self._embed_model = index._embed_model

        self._service_context = self._index.service_context
        self._docstore = self._index.docstore

        self._similarity_top_k = similarity_top_k
        self._image_similarity_top_k = image_similarity_top_k
        self._vector_store_query_mode = VectorStoreQueryMode(vector_store_query_mode)
        self._alpha = alpha
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._filters = filters
        self._sparse_top_k = sparse_top_k
        self._search_image = self._enable_multimodal & search_image

        # Hybrid search parameters
        self._supports_hybrid_search = supports_hybrid_search
        self._local_bm25_index = local_bm25_index
        if not hybrid_fusion_weights:
            self._hybrid_fusion_weights = [0.5, 0.5]
        else:
            # Sum of retriever_weights must be 1
            total_weight = sum(hybrid_fusion_weights)
            self._hybrid_fusion_weights = [
                w / total_weight for w in hybrid_fusion_weights
            ]

        self._kwargs: Dict[str, Any] = kwargs.get("vector_store_kwargs", {})
        self.callback_manager = (
            callback_manager
            or callback_manager_from_settings_or_context(
                Settings, self._service_context
            )
        )

    @property
    def similarity_top_k(self) -> int:
        """Return similarity top k."""
        return self._similarity_top_k

    @similarity_top_k.setter
    def similarity_top_k(self, similarity_top_k: int) -> None:
        """Set similarity top k."""
        self._similarity_top_k = similarity_top_k

    @property
    def image_similarity_top_k(self) -> int:
        """Return image similarity top k."""
        return self._image_similarity_top_k

    @image_similarity_top_k.setter
    def image_similarity_top_k(self, image_similarity_top_k: int) -> None:
        """Set image similarity top k."""
        self._image_similarity_top_k = image_similarity_top_k

    def _build_vector_store_query(
        self,
        query_bundle_with_embeddings: QueryBundle,
        similarity_top_k: int,
        is_image_store: bool = False,
    ) -> VectorStoreQuery:
        query_mode = self._vector_store_query_mode

        if is_image_store:
            # image store does not apply hybrid/keyword search
            query_mode = VectorStoreQueryMode.DEFAULT

        return VectorStoreQuery(
            query_embedding=query_bundle_with_embeddings.embedding,
            similarity_top_k=similarity_top_k,
            node_ids=self._node_ids,
            doc_ids=self._doc_ids,
            query_str=query_bundle_with_embeddings.query_str,
            mode=query_mode,
            alpha=self._alpha,
            filters=self._filters,
            sparse_top_k=self._sparse_top_k,
        )

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        text_nodes, image_nodes = [], []
        # If text vector store is not empty, retrieve text nodes
        # If text vector store is empty, please create index without text vector store
        if self._vector_store is not None:
            text_nodes = self._text_retrieve(query_bundle)

        # If image vector store is not empty, retrieve text nodes
        # If image vector store is empty, please create index without image vector store
        if self._enable_multimodal and self._image_vector_store is not None:
            image_nodes = self._text_to_image_retrieve(query_bundle)

        seen_images = set([node.node.image_url for node in image_nodes])
        # 从文本中召回图片
        if self._search_image and len(image_nodes) < self._image_similarity_top_k:
            for node in text_nodes:
                image_url_infos = node.node.metadata.get("image_info_list")
                if not image_url_infos:
                    continue
                for image_url_info in image_url_infos:
                    if image_url_info.get("image_url", None) not in seen_images:
                        image_nodes.extend(
                            NodeWithScore(
                                ImageNode(
                                    image_url=image_url_info.get("image_url", None)
                                ),
                                score=node.score
                                * 0.5,  # discount the score from text nodes
                            )
                        )
                        seen_images.add(image_url_info.get("image_url", None))
                if len(image_nodes) >= self._image_similarity_top_k:
                    break

        if not text_nodes:
            text_nodes = []
        if not image_nodes:
            image_nodes = []
        results = text_nodes + image_nodes
        return results

    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Retrieve nodes given query.

        Args:
            str_or_query_bundle (QueryType): Either a query string or
                a QueryBundle object.

        """
        dispatch_event = dispatcher.get_dispatch_event()

        self._check_callback_manager()
        dispatch_event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self._retrieve(query_bundle)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatch_event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    def _text_retrieve_from_vector_store(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                query_bundle.embedding = (
                    self._embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )

        return self._get_nodes_with_embeddings(
            query_bundle, self._similarity_top_k, self._vector_store
        )

    async def _atext_retrieve_from_vector_store(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                query_bundle.embedding = (
                    await self._embed_model.aget_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )

        return await self._aget_nodes_with_embeddings(
            query_bundle, self._similarity_top_k, self._vector_store
        )

    def _fusion_nodes(
        self,
        vector_nodes: List[NodeWithScore],
        keyword_nodes: List[NodeWithScore],
        similarity_top_k: int,
    ):
        # print("Fusion weights: ", self._hybrid_fusion_weights)

        for node_with_score in vector_nodes:
            # print("vector score 0", node_with_score.node_id, node_with_score.score)
            node_with_score.score *= self._hybrid_fusion_weights[0]
            # print("vector score 1", node_with_score.node_id, node_with_score.score)

        for node_with_score in keyword_nodes:
            # print("keyword score 0", node_with_score.node_id, node_with_score.score)
            node_with_score.score *= self._hybrid_fusion_weights[1]
            # print("keyword score 1", node_with_score.node_id, node_with_score.score)

        # Use a dict to de-duplicate nodes
        all_nodes: Dict[str, NodeWithScore] = {}

        # Sum scores for each node
        for nodes_with_scores in [vector_nodes, keyword_nodes]:
            for node_with_score in nodes_with_scores:
                key = node_with_score.node.node_id
                if key in all_nodes:
                    all_nodes[key].score += node_with_score.score
                else:
                    all_nodes[key] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)[
            :similarity_top_k
        ]

    def _text_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._similarity_top_k <= 0:
            return []
        if (
            self._supports_hybrid_search
            or self._vector_store_query_mode == VectorStoreQueryMode.DEFAULT
        ):
            return self._text_retrieve_from_vector_store(query_bundle)
        elif (
            self._vector_store_query_mode == VectorStoreQueryMode.TEXT_SEARCH
            or self._vector_store_query_mode == VectorStoreQueryMode.SPARSE
        ):
            return self._local_bm25_index.query(
                query_str=query_bundle.query_str,
                top_n=self.similarity_top_k,
                normalize=True,
            )
        else:
            vector_nodes = self._text_retrieve_from_vector_store(query_bundle)
            keyword_nodes = self._local_bm25_index.query(
                query_str=query_bundle.query_str,
                top_n=self.similarity_top_k,
                normalize=True,
            )

            return self._fusion_nodes(
                vector_nodes, keyword_nodes, self._similarity_top_k
            )

    def text_retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._text_retrieve(str_or_query_bundle)

    def _text_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if (
            self._image_vector_store.is_embedding_query
            and self._image_similarity_top_k > 0
        ):
            # change the embedding for query bundle to Multi Modal Text encoder
            query_bundle.embedding = (
                self._image_embed_model.get_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
            )

            return self._get_nodes_with_embeddings(
                query_bundle, self._image_similarity_top_k, self._image_vector_store
            )
        else:
            return []

    def text_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._text_to_image_retrieve(str_or_query_bundle)

    def _image_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._image_vector_store.is_embedding_query:
            # change the embedding for query bundle to Multi Modal Image encoder for image input
            assert isinstance(self._index.image_embed_model, MultiModalEmbedding)
            query_bundle.embedding = self._image_embed_model.get_image_embedding(
                query_bundle.embedding_image[0]
            )
        return self._get_nodes_with_embeddings(
            query_bundle, self._image_similarity_top_k, self._image_vector_store
        )

    def image_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(
                query_str="", image_path=str_or_query_bundle
            )
        return self._image_to_image_retrieve(str_or_query_bundle)

    def _get_nodes_with_embeddings(
        self,
        query_bundle_with_embeddings: QueryBundle,
        similarity_top_k: int,
        vector_store: BasePydanticVectorStore,
    ) -> List[NodeWithScore]:
        is_image_store = (
            self._image_vector_store and vector_store is self._image_vector_store
        )
        query = self._build_vector_store_query(
            query_bundle_with_embeddings, similarity_top_k, is_image_store
        )
        query_result = vector_store.query(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result, is_image_store)

    def _build_node_list_from_query_result(
        self,
        query_result: VectorStoreQueryResult,
        is_image=False,
    ) -> List[NodeWithScore]:
        if query_result.nodes is None:
            # NOTE: vector store does not keep text and returns node indices.
            # Need to recover all nodes from docstore, i.e. FaissVectorStore
            if query_result.ids is None:
                raise ValueError(
                    "Vector store query result should return at "
                    "least one of nodes or ids."
                )
            assert isinstance(self._index.index_struct, IndexDict)
            node_ids = [
                self._index.index_struct.nodes_dict[
                    f"{DEFAULT_IMAGE_STORE}_{idx}" if is_image else f"{idx}"
                ]
                for idx in query_result.ids
            ]
            nodes = self._docstore.get_nodes(node_ids)
            query_result.nodes = nodes
        else:
            # NOTE: vector store keeps text, returns nodes.
            # Only need to recover image or index nodes from docstore, i.e. Hologres
            for i in range(len(query_result.nodes)):
                node = query_result.nodes[i]
                node.embedding = None
                if is_image:
                    node = ImageNode(
                        id_=node.id_,
                        image_url=node.metadata.get("image_url"),
                        metadata=node.metadata,
                    )
                query_result.nodes[i] = node
        # else:
        #     # NOTE: vector store keeps text, returns nodes.
        #     # Only need to recover image or index nodes from docstore
        #     for i in range(len(query_result.nodes)):
        #         source_node = query_result.nodes[i].source_node
        #         if (not self._vector_store.stores_text) or (
        #             source_node is not None and source_node.node_type != ObjectType.TEXT
        #         ):
        #             node_id = query_result.nodes[i].node_id
        #             if self._docstore.document_exists(node_id):
        #                 query_result.nodes[
        #                     i
        #                 ] = self._docstore.get_node(  # type: ignore[index]
        #                     node_id
        #                 )

        log_vector_store_query_result(query_result)

        node_with_scores: List[NodeWithScore] = []
        for ind, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
            node_with_scores.append(NodeWithScore(node=node, score=score))
        return node_with_scores

    # Async Retrieval Methods

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Run the two retrievals in async, and return their results as a concatenated list
        results: List[NodeWithScore] = []
        tasks = [
            self._atext_retrieve(query_bundle),
            self._atext_to_image_retrieve(query_bundle),
        ]
        task_results = await asyncio.gather(*tasks)

        text_nodes, image_nodes = task_results[0], task_results[1]
        logger.info(f"Retrieved text nodes: {text_nodes}")
        logger.info(f"Retrieved image nodes: {image_nodes}")

        seen_images = set([node.node.image_url for node in image_nodes])
        # 从文本中召回图片
        if self._search_image and len(image_nodes) < self._image_similarity_top_k:
            for node in text_nodes:
                image_url_infos = node.node.metadata.get("image_info_list")
                if not image_url_infos:
                    continue
                for image_url_info in image_url_infos:
                    if image_url_info.get("image_url", None) not in seen_images:
                        image_nodes.extend(
                            NodeWithScore(
                                ImageNode(
                                    image_url=image_url_info.get("image_url", None)
                                ),
                                score=node.score
                                * 0.5,  # discount the score from text nodes
                            )
                        )
                        seen_images.add(image_url_info.get("image_url", None))
                if len(image_nodes) >= self._image_similarity_top_k:
                    break

        if not text_nodes:
            text_nodes = []
        if not image_nodes:
            image_nodes = []
        results = text_nodes + image_nodes
        return results

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        self._check_callback_manager()
        dispatch_event = dispatcher.get_dispatch_event()

        dispatch_event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(query_bundle=query_bundle)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatch_event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    async def _atext_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._similarity_top_k <= 0:
            return []
        if (
            self._supports_hybrid_search
            or self._vector_store_query_mode == VectorStoreQueryMode.DEFAULT
        ):
            return await self._atext_retrieve_from_vector_store(query_bundle)
        elif (
            self._vector_store_query_mode == VectorStoreQueryMode.TEXT_SEARCH
            or self._vector_store_query_mode == VectorStoreQueryMode.SPARSE
        ):
            return self._local_bm25_index.query(
                query_str=query_bundle.query_str,
                top_n=self.similarity_top_k,
                normalize=True,
            )
        else:
            vector_nodes = await self._atext_retrieve_from_vector_store(query_bundle)
            keyword_nodes = self._local_bm25_index.query(
                query_str=query_bundle.query_str,
                top_n=self.similarity_top_k,
                normalize=True,
            )

            return self._fusion_nodes(
                vector_nodes, keyword_nodes, self.similarity_top_k
            )

    async def atext_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._atext_retrieve(str_or_query_bundle)

    async def _atext_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if (
            self._enable_multimodal
            and self._search_image
            and self._image_similarity_top_k > 0
        ):
            if self._image_vector_store.is_embedding_query:
                # change the embedding for query bundle to Multi Modal Text encoder
                query_bundle.embedding = (
                    await self._image_embed_model.aget_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )

            return await self._aget_nodes_with_embeddings(
                query_bundle, self._image_similarity_top_k, self._image_vector_store
            )
        else:
            return []

    async def atext_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return await self._atext_to_image_retrieve(str_or_query_bundle)

    async def _aget_nodes_with_embeddings(
        self,
        query_bundle_with_embeddings: QueryBundle,
        similarity_top_k: int,
        vector_store: BasePydanticVectorStore,
    ) -> List[NodeWithScore]:
        is_image_store = (
            self._image_vector_store and vector_store is self._image_vector_store
        )
        query = self._build_vector_store_query(
            query_bundle_with_embeddings, similarity_top_k, is_image_store
        )
        query_result = await vector_store.aquery(query, **self._kwargs)

        return self._build_node_list_from_query_result(query_result, is_image_store)

    async def _aimage_to_image_retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._image_vector_store.is_embedding_query:
            # change the embedding for query bundle to Multi Modal Image encoder for image input
            assert isinstance(self._index.image_embed_model, MultiModalEmbedding)
            # Using the first imaage in the list for image retrieval
            query_bundle.embedding = await self._image_embed_model.aget_image_embedding(
                query_bundle.embedding_image[0]
            )
        return await self._aget_nodes_with_embeddings(
            query_bundle, self._image_similarity_top_k, self._image_vector_store
        )

    async def aimage_to_image_retrieve(
        self, str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            # leave query_str as empty since we are using image_path for image retrieval
            str_or_query_bundle = QueryBundle(
                query_str="", image_path=str_or_query_bundle
            )
        return await self._aimage_to_image_retrieve(str_or_query_bundle)
