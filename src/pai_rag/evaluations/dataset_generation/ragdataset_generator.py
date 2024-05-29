from llama_index.core.llama_dataset.base import BaseLlamaDataExample
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.bridge.pydantic import Field

import re
from typing import List, Optional

from llama_index.core import Document, SummaryIndex
from llama_index.core.async_utils import run_jobs, asyncio_run
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabelledRagDataset,
)

from llama_index.core.schema import BaseNode


class ModifiedRagDatasetGenerator(RagDatasetGenerator):
    async def _agenerate_dataset(
        self,
        nodes: List[BaseNode],
        labelled: bool = False,
    ) -> LabelledRagDataset:
        """Node question generator."""
        query_tasks = []
        examples: List[LabelledRagDataExample] = []
        summary_indices: List[SummaryIndex] = []
        for node in nodes:
            index = SummaryIndex.from_documents(
                [
                    Document(
                        text=node.get_content(metadata_mode=self._metadata_mode),
                        metadata=node.metadata,
                        excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                        excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                        relationships=node.relationships,
                    )
                ],
            )

            query_engine = index.as_query_engine(
                llm=self._llm,
                text_qa_template=self.text_question_template,
                use_async=True,
            )
            task = query_engine.aquery(
                self.question_gen_query,
            )
            query_tasks.append(task)
            summary_indices.append(index)

        responses = await run_jobs(query_tasks, self._show_progress, self._workers)
        for idx, response in enumerate(responses):
            result = str(response).strip().split("\n")
            cleaned_questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            cleaned_questions = [
                question for question in cleaned_questions if len(question) > 0
            ]
            index = summary_indices[idx]
            reference_context = nodes[idx].text
            model_name = self._llm.metadata.model_name
            created_by = CreatedBy(type=CreatedByType.AI, model_name=model_name)
            if labelled:
                index = summary_indices[idx]
                qr_tasks = []
                for query in cleaned_questions:
                    # build summary index off of node (i.e. context)
                    qa_query_engine = index.as_query_engine(
                        llm=self._llm,
                        text_qa_template=self.text_qa_template,
                    )
                    qr_task = qa_query_engine.aquery(query)
                    qr_tasks.append(qr_task)
                answer_responses: List[RESPONSE_TYPE] = await run_jobs(
                    qr_tasks, self._show_progress, self._workers
                )
                for (
                    question,
                    answer_response,
                ) in zip(cleaned_questions, answer_responses):
                    example = LabelledRagDataExample(
                        query=question,
                        reference_answer=str(answer_response),
                        reference_contexts=[reference_context],
                        reference_node_id=[nodes[idx].node_id],
                        reference_answer_by=created_by,
                        query_by=created_by,
                    )
                    examples.append(example)
            else:
                for query in cleaned_questions:
                    example = LabelledRagDataExample(
                        query=query,
                        reference_answer="",
                        reference_contexts=[reference_context],
                        reference_answer_by=None,
                        query_by=created_by,
                    )
                    examples.append(example)

        # split train/test
        return LabelledRagDataset(examples=examples)

    async def agenerate_questions_from_nodes(self) -> LabelledRagDataset:
        """Generates questions but not the reference answers."""
        return await self._agenerate_dataset(self.nodes, labelled=False)

    async def agenerate_dataset_from_nodes(self) -> LabelledRagDataset:
        """Generates questions for each document."""
        return await self._agenerate_dataset(self.nodes, labelled=True)

    def generate_questions_from_nodes(self) -> LabelledRagDataset:
        """Generates questions but not the reference answers."""
        return asyncio_run(self.agenerate_questions_from_nodes())

    def generate_dataset_from_nodes(self) -> LabelledRagDataset:
        """Generates questions for each document."""
        return asyncio_run(self.agenerate_dataset_from_nodes())


class LabelledRagDataExample(BaseLlamaDataExample):
    """RAG example class. Analogous to traditional ML datasets, this dataset contains
    the "features" (i.e., query + context) to make a prediction and the "label" (i.e., response)
    to evaluate the prediction.

    Args:
        query (str): The user query
        query_by (CreatedBy): Query generated by human or ai (model-name)
        reference_contexts (Optional[List[str]]): The contexts used for response
        reference_node_id (Optional[List[str]]): The node id corresponding to the contexts
        reference_answer ([str]): Reference answer to the query. An answer
                                    that would receive full marks upon evaluation.
        reference_answer_by: The reference answer generated by human or ai (model-name).
    """

    query: str = Field(
        default_factory=str, description="The user query for the example."
    )
    query_by: Optional[CreatedBy] = Field(
        default=None, description="What generated the query."
    )
    reference_contexts: Optional[List[str]] = Field(
        default_factory=None,
        description="The contexts used to generate the reference answer.",
    )
    reference_node_id: Optional[List[str]] = Field(
        default_factory=None, description="The node id corresponding to the contexts"
    )
    reference_answer: str = Field(
        default_factory=str,
        description="The reference (ground-truth) answer to the example.",
    )
    reference_answer_by: Optional[CreatedBy] = Field(
        default=None, description="What generated the reference answer."
    )

    @property
    def class_name(self) -> str:
        """Data example class name."""
        return "LabelledRagDataExample"
