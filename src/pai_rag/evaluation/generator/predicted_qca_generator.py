from typing import List
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.async_utils import run_jobs
from pai_rag.evaluation.generator.rag_qca_sample import PredictedRagQcaSample
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabelledRagDataset,
)
from pai_rag.evaluation.generator.utils import (
    load_qca_dataset_json,
    save_qca_dataset_json,
)
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

    async def agenerate_predicted_qca_dataset(
        self,
    ):
        predicted_qca_dataset_path = os.path.join(
            self.persist_path, "predicted_qca_dataset.json"
        )
        if os.path.exists(predicted_qca_dataset_path):
            print(
                f"A predicted QCA dataset already exists at {predicted_qca_dataset_path}."
            )
            return load_qca_dataset_json(predicted_qca_dataset_path)
        else:
            print("Starting to generate predicted QCA dataset...")
            labelled_qca_dataset_path = os.path.join(
                self.persist_path, "labelled_qca_dataset.json"
            )
            labelled_qca_dataset = load_qca_dataset_json(labelled_qca_dataset_path)
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
                    predicted_contexts=[
                        node.node.text for node in response.source_nodes
                    ],
                    predicted_node_id=[
                        node.node.node_id for node in response.source_nodes
                    ],
                    predicted_answer=response.response,
                    predicted_answer_by=self.created_by,
                )
                examples.append(example)
            predicted_qca_dataset = LabelledRagDataset(examples=examples)
            save_qca_dataset_json(predicted_qca_dataset, predicted_qca_dataset_path)
            return predicted_qca_dataset
