from typing import List, Optional

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
from llama_index.core import VectorStoreIndex
from pai_rag.integrations.retrievers.raptor.raptor_clusters import get_clusters

import logging

logger = logging.getLogger(__name__)


DEFAULT_SUMMARY_PROMPT = (
    "Summarize the provided text in Chinese, including as many key details as needed."
)


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

    async def generate_summaries(
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


async def enhance_nodes(
    nodes: List[BaseNode],
    index: VectorStoreIndex,
    tree_depth: int = 2,
) -> VectorStoreIndex:
    """Given a set of nodes, this function inserts higher level of abstractions within the index.

    For later retrieval

    Args:
        nodes (List[BaseNode]): List of nodes
        index: created VectorStoreIndex
        tree_depth: usually 2 or 3
    """

    embed_model = index._embed_model
    summary_module = SummaryModule()

    cur_nodes = nodes
    for level in range(tree_depth):
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
        nodes_per_cluster = get_clusters(cur_nodes, id_to_embedding)

        logger.info(
            f"Generating summaries for level {level} with {len(nodes_per_cluster)} clusters."
        )
        summaries_per_cluster = await summary_module.generate_summaries(
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

        # insert the nodes with their embeddings and parent_id
        nodes_with_embeddings = []
        for cluster, summary_doc in zip(nodes_per_cluster, new_nodes):
            for node in cluster:
                node.metadata["parent_id"] = summary_doc.id_
                node.excluded_embed_metadata_keys.append("parent_id")
                node.excluded_llm_metadata_keys.append("parent_id")
                node.embedding = id_to_embedding[node.id_]
                nodes_with_embeddings.append(node)

        index.insert_nodes(nodes_with_embeddings)

        # set the current nodes to the new nodes
        cur_nodes = new_nodes

    index.insert_nodes(cur_nodes)

    return index
