from typing import List, Optional, Any

import asyncio

from llama_index.core import get_response_synthesizer
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms.llm import LLM
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import (
    BaseNode,
    NodeWithScore,
    TextNode,
)
from llama_index.core.schema import TransformComponent
from pai_rag.integrations.nodes.raptor_clusters import get_clusters
from pai_rag.utils.prompt_template import DEFAULT_SUMMARY_PROMPT

import logging

logger = logging.getLogger(__name__)


class RaptorProcessor(TransformComponent):
    tree_depth: int
    max_clusters: int
    threshold: float
    embed_model: Any

    def __init__(
        self, tree_depth: int, max_clusters: int, threshold: float, embed_model: Any
    ) -> None:
        """get params from config"""
        super().__init__(
            tree_depth=tree_depth,
            max_clusters=max_clusters,
            threshold=threshold,
            embed_model=embed_model,
        )

    def __call__(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Given a set of nodes, this function inserts higher level of abstractions within the index.

        For later retrieval

        Args:
            nodes (List[BaseNode]): List of nodes
            index: created VectorStoreIndex
            tree_depth: usually 2 or 3
            max_length_in_cluster: max token number in a cluster
            max_clusters: max number of tokens set for BIC
            threshold: probability threshold
        """

        embed_model = self.embed_model
        summary_module = SummaryModule()

        cur_nodes = nodes
        new_nodes_collection = []
        nodes_with_embeddings_collections = []
        for level in range(self.tree_depth):
            # get the embeddings for the current documents

            logger.info(f"Generating embeddings for level {level}.")

            embeddings = embed_model.get_text_embedding_batch(
                [node.get_content(metadata_mode="embed") for node in cur_nodes]
            )
            assert len(embeddings) == len(cur_nodes)
            id_to_embedding = {
                node.id_: embedding for node, embedding in zip(cur_nodes, embeddings)
            }

            logger.info(f"Performing clustering for level {level}.")

            # cluster the documents
            nodes_per_cluster = get_clusters(
                nodes=cur_nodes,
                embedding_map=id_to_embedding,
                max_clusters=self.max_clusters,
                threshold=self.threshold,
            )

            logger.info(
                f"Generating summaries for level {level} with {len(nodes_per_cluster)} clusters."
            )
            summaries_per_cluster = summary_module.generate_summaries(
                documents_per_cluster=nodes_per_cluster
            )

            logger.info(
                f"Level {level} created summaries/clusters: {len(nodes_per_cluster)}"
            )

            # replace the current nodes with their summaries
            new_nodes = [
                TextNode(
                    text=summary,
                    metadata={"level": level},
                    excluded_embed_metadata_keys=["level"],
                    excluded_llm_metadata_keys=["level"],
                )
                for summary in summaries_per_cluster
            ]
            new_nodes_collection.extend(new_nodes)

            # insert the nodes with their embeddings and parent_id
            nodes_with_embeddings = []
            for cluster, summary_doc in zip(nodes_per_cluster, new_nodes):
                for node in cluster:
                    node.metadata["parent_id"] = summary_doc.id_
                    node.excluded_embed_metadata_keys.append("parent_id")
                    node.excluded_llm_metadata_keys.append("parent_id")
                    node.embedding = id_to_embedding[node.id_]
                    nodes_with_embeddings.append(node)

            nodes_with_embeddings_collections.append(nodes_with_embeddings)

            # set the current nodes to the new nodes
            cur_nodes = new_nodes

            if level == self.tree_depth - 1:
                embeddings = embed_model.get_text_embedding_batch(
                    [node.get_content(metadata_mode="embed") for node in cur_nodes]
                )
                assert len(embeddings) == len(cur_nodes)
                id_to_embedding = {
                    node.id_: embedding
                    for node, embedding in zip(cur_nodes, embeddings)
                }
                for node in cur_nodes:
                    node.metadata["parent_id"] = ""
                    node.excluded_embed_metadata_keys.append("parent_id")
                    node.excluded_llm_metadata_keys.append("parent_id")
                    node.embedding = id_to_embedding[node.id_]
                    nodes_with_embeddings_collections.append([node])

        nodes_with_embeddings_collections = [
            k for i in nodes_with_embeddings_collections for k in i
        ]

        return nodes_with_embeddings_collections

    async def acall(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        """Async transform nodes."""

        embed_model = self.embed_model
        summary_module = SummaryModule()

        cur_nodes = nodes
        new_nodes_collection = []
        nodes_with_embeddings_collections = []
        for level in range(self.tree_depth):
            # get the embeddings for the current documents

            logger.info(f"Generating embeddings for level {level}.")

            embeddings = await embed_model.aget_text_embedding_batch(
                [node.get_content(metadata_mode="embed") for node in cur_nodes]
            )
            assert len(embeddings) == len(cur_nodes)
            id_to_embedding = {
                node.id_: embedding for node, embedding in zip(cur_nodes, embeddings)
            }

            logger.info(f"Performing clustering for level {level}.")

            # cluster the documents
            nodes_per_cluster = get_clusters(
                nodes=cur_nodes,
                embedding_map=id_to_embedding,
                max_clusters=self.max_clusters,
                threshold=self.threshold,
            )

            logger.info(
                f"Generating summaries for level {level} with {len(nodes_per_cluster)} clusters."
            )
            summaries_per_cluster = await summary_module.agenerate_summaries(
                documents_per_cluster=nodes_per_cluster
            )

            logger.info(
                f"Level {level} created summaries/clusters: {len(nodes_per_cluster)}"
            )

            # replace the current nodes with their summaries
            new_nodes = [
                TextNode(
                    text=summary,
                    metadata={"level": level},
                    excluded_embed_metadata_keys=["level"],
                    excluded_llm_metadata_keys=["level"],
                )
                for summary in summaries_per_cluster
            ]
            new_nodes_collection.extend(new_nodes)

            # insert the nodes with their embeddings and parent_id
            nodes_with_embeddings = []
            for cluster, summary_doc in zip(nodes_per_cluster, new_nodes):
                for node in cluster:
                    node.metadata["parent_id"] = summary_doc.id_
                    node.excluded_embed_metadata_keys.append("parent_id")
                    node.excluded_llm_metadata_keys.append("parent_id")
                    node.embedding = id_to_embedding[node.id_]
                    nodes_with_embeddings.append(node)

            nodes_with_embeddings_collections.append(nodes_with_embeddings)

            # set the current nodes to the new nodes
            cur_nodes = new_nodes

            if level == self.tree_depth - 1:
                embeddings = await embed_model.aget_text_embedding_batch(
                    [node.get_content(metadata_mode="embed") for node in cur_nodes]
                )
                assert len(embeddings) == len(cur_nodes)
                id_to_embedding = {
                    node.id_: embedding
                    for node, embedding in zip(cur_nodes, embeddings)
                }
                for node in cur_nodes:
                    node.metadata["parent_id"] = ""
                    node.excluded_embed_metadata_keys.append("parent_id")
                    node.excluded_llm_metadata_keys.append("parent_id")
                    node.embedding = id_to_embedding[node.id_]
                    nodes_with_embeddings_collections.append([node])

        nodes_with_embeddings_collections = [
            k for i in nodes_with_embeddings_collections for k in i
        ]

        return nodes_with_embeddings_collections


class SummaryModule(BaseModel):
    response_synthesizer: BaseSynthesizer = Field(description="LLM")
    summary_prompt: str = Field(
        default=DEFAULT_SUMMARY_PROMPT,
        description="Summary prompt.",
    )
    num_workers: int = Field(
        default=4, description="Number of workers to generate summaries."
    )
    show_progress: bool = Field(default=True, description="Show progress.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        llm: Optional[LLM] = None,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        num_workers: int = 4,
    ) -> None:
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", use_async=True, llm=llm
        )
        super().__init__(
            response_synthesizer=response_synthesizer,
            summary_prompt=summary_prompt,
            num_workers=num_workers,
        )

    def generate_summaries(
        self, documents_per_cluster: List[List[BaseNode]]
    ) -> List[str]:
        """Generate summaries of documents per cluster.

        Args:
            documents_per_cluster (List[List[BaseNode]]): List of documents per cluster

        Returns:
            List[str]: List of summary for each cluster
        """
        responses = []
        for documents in documents_per_cluster:
            with_scores = [NodeWithScore(node=doc, score=1.0) for doc in documents]
            responses.append(
                self.response_synthesizer.synthesize(self.summary_prompt, with_scores)
            )
        return [str(response) for response in responses]

    async def agenerate_summaries(
        self, documents_per_cluster: List[List[BaseNode]]
    ) -> List[str]:
        """Generate summaries of documents per cluster.

        Args:
            documents_per_cluster (List[List[BaseNode]]): List of documents per cluster

        Returns:
            List[str]: List of summary for each cluster
        """
        jobs = []
        for documents in documents_per_cluster:
            with_scores = [NodeWithScore(node=doc, score=1.0) for doc in documents]
            jobs.append(
                self.response_synthesizer.asynthesize(self.summary_prompt, with_scores)
            )

        lock = asyncio.Semaphore(self.num_workers)
        responses = []

        # run the jobs while limiting the number of concurrent jobs to num_workers
        for job in jobs:
            async with lock:
                responses.append(await job)

        return [str(response) for response in responses]
