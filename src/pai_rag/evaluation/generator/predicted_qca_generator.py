from typing import List
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.async_utils import run_jobs
from pai_rag.evaluation.generator.rag_qca_sample import PredictedRagQcaSample
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabelledRagDataset,
)

import json
import os
import logging

logger = logging.getLogger(__name__)


class PredictedRagQcaGenerator:
    def __init__(
        self, llm, vector_index: VectorStoreIndex = None, persist_path: str = None
    ):
        self._vector_index = vector_index._vector_index
        self._llm = llm
        model_name = self._llm.metadata.model_name
        self.created_by = CreatedBy(type=CreatedByType.AI, model_name=model_name)
        self._query_engine = vector_index._vector_index.as_query_engine(llm=self._llm)
        self.persist_path = persist_path
        self._show_progress = True
        self._workers = 2

    def from_labelled_qca_dataset(self) -> LabelledRagDataset:
        """Load json."""
        labelled_qca_dataset_path = os.path.join(
            self.persist_path, "labelled_qca_dataset.json"
        )
        with open(labelled_qca_dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        return data

    def save_predicted_qca_dataset_json(self, qas: LabelledRagDataset) -> None:
        """Save json."""
        predicted_qca_dataset_path = os.path.join(
            self.persist_path, "predicted_qca_dataset.json"
        )
        with open(predicted_qca_dataset_path, "w", encoding="utf-8") as f:
            json.dump(qas.dict(), f, indent=4, ensure_ascii=False)
        logger.info(f"Saved labelled qca dataset to {predicted_qca_dataset_path}")

    async def agenerate_predicted_qca_dataset(
        self,
    ):
        labelled_qca_dataset = self.from_labelled_qca_dataset()
        query_tasks = []
        labelled_qca_set = []
        for qca in labelled_qca_dataset["examples"]:
            task = self._query_engine.aquery(qca["query"])
            labelled_qca_set.append(qca)
            query_tasks.append(task)
        responses = await run_jobs(query_tasks, self._show_progress, self._workers)
        examples: List[PredictedRagQcaSample] = []
        for qca, response in zip(labelled_qca_set, responses):
            example = PredictedRagQcaSample(
                query=qca["query"],
                reference_answer=qca["reference_answer"],
                reference_contexts=qca["reference_contexts"],
                reference_node_id=qca["reference_node_id"],
                reference_answer_by=qca["reference_answer_by"],
                query_by=qca["query_by"],
                predicted_contexts=[node.node.text for node in response.source_nodes],
                predicted_node_id=[node.node.node_id for node in response.source_nodes],
                predicted_answer=response.response,
                predicted_answer_by=self.created_by,
            )
            examples.append(example)
        predicted_qca_dataset = LabelledRagDataset(examples=examples)
        self.save_predicted_qca_dataset_json(predicted_qca_dataset)
        return predicted_qca_dataset
