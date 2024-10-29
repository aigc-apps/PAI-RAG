from typing import List
from llama_index.core.indices import VectorStoreIndex
from pai_rag.utils.prompt_template import (
    DEFAULT_QUESTION_GENERATION_PROMPT,
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_QUESTION_GENERATION_QUERY,
)
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.prompts.base import PromptTemplate
import re
from llama_index.core.async_utils import run_jobs
from pai_rag.evaluation.dataset.rag_qca_dataset import RagQcaSample, PaiRagQcaDataset
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
)
from pai_rag.integrations.synthesizer.pai_synthesizer import PaiQueryBundle

import os
import logging
from pai_rag.integrations.query_engine.pai_retriever_query_engine import (
    PaiRetrieverQueryEngine,
)
from pai_rag.integrations.llms.pai.pai_multi_modal_llm import (
    PaiMultiModalLlm,
)

logger = logging.getLogger(__name__)


class RagQcaGenerator:
    def __init__(
        self,
        llm,
        vector_index: VectorStoreIndex = None,
        query_engine: PaiRetrieverQueryEngine = None,
        persist_path: str = None,
    ):
        self._llm = llm
        self._vector_index = vector_index._vector_index
        self._query_engine = query_engine
        self.question_gen_query = DEFAULT_QUESTION_GENERATION_QUERY.format(
            num_questions_per_chunk=3
        )
        self.text_question_template = PromptTemplate(DEFAULT_QUESTION_GENERATION_PROMPT)
        self.text_question_answer_template = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)
        self.created_by = CreatedBy(
            type=CreatedByType.AI, model_name=self._llm.metadata.model_name
        )
        self.persist_path = persist_path
        self.qca_dataset_path = os.path.join(self.persist_path, "qca_dataset.json")
        self._show_progress = True
        self._workers = 2

    def load_qca_dataset(self) -> None:
        if os.path.exists(self.qca_dataset_path):
            rag_qca_dataset = PaiRagQcaDataset.from_json(self.qca_dataset_path)
            print(
                f"A RAG QCA dataset already exists at {self.qca_dataset_path} with status: [labelled: {rag_qca_dataset.labelled}, predicted: {rag_qca_dataset.predicted}]."
            )
            return rag_qca_dataset
        else:
            print("No existing QCA dataset found. You can proceed to create a new one.")
            return None

    async def agenerate_qca_dataset(self, stage):
        rag_qca_dataset = self.load_qca_dataset()
        if rag_qca_dataset and rag_qca_dataset.labelled:
            if stage == "labelled":
                print("Labelled QCA dataset already exists. Skipping labelled stage.")
                return rag_qca_dataset.examples
            elif stage == "predicted":
                if rag_qca_dataset.predicted:
                    print(
                        "Predicted QCA dataset already exists. Skipping predicted stage."
                    )
                    return rag_qca_dataset.examples
                else:
                    return await self.agenerate_predicted_qca_dataset(rag_qca_dataset)
            else:
                raise ValueError(f"Invalid stage: {stage}")
        else:
            return await self.agenerate_labelled_qca_dataset()

    async def agenerate_labelled_qca_sample(self, node):
        prompt_str = self.text_question_template.format(
            context_str=node.text, num_questions_per_chunk=1
        )
        if isinstance(self._llm, PaiMultiModalLlm):
            response = await self._llm.acomplete(
                prompt=prompt_str, image_documents=None
            )
        else:
            response = await self._llm.acomplete(prompt=prompt_str)
        result = str(response).strip().split("\n")
        cleaned_questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        cleaned_questions = [
            question for question in cleaned_questions if len(question) > 0
        ]
        qr_tasks = []
        for query in cleaned_questions:
            prompt_str = self.text_question_answer_template.format(
                context_str=node.text, query_str=query
            )
            if isinstance(self._llm, PaiMultiModalLlm):
                qr_task = self._llm.acomplete(prompt=prompt_str, image_documents=None)
            else:
                qr_task = self._llm.acomplete(prompt=prompt_str)
            qr_tasks.append(qr_task)
        answer_responses: List[RESPONSE_TYPE] = await run_jobs(
            qr_tasks, self._show_progress, self._workers
        )
        for (
            question,
            answer_response,
        ) in zip(cleaned_questions, answer_responses):
            sample = RagQcaSample(
                query=question,
                reference_answer=str(answer_response),
                reference_contexts=[node.text],
                reference_node_id=[node.node_id],
                reference_answer_by=self.created_by,
                query_by=self.created_by,
            )
            return sample

    async def agenerate_labelled_qca_dataset(
        self,
    ):
        print("Starting to generate QCA dataset for [[labelled]].")
        docs = self._vector_index._docstore.docs
        nodes = list(docs.values())
        tasks = []
        for node in nodes:
            tasks.append(self.agenerate_labelled_qca_sample(node))
        examples = await run_jobs(tasks, self._show_progress, self._workers)
        labelled_qca_dataset = PaiRagQcaDataset(examples=examples, labelled=True)
        labelled_qca_dataset.save_json(self.qca_dataset_path)
        return labelled_qca_dataset.examples

    async def agenerate_predicted_qca_sample(self, qca_sample):
        query_bundle = PaiQueryBundle(query_str=qca_sample.query)
        response = await self._query_engine.aquery(query_bundle)
        qca_sample.predicted_answer = response.response
        qca_sample.predicted_contexts = [
            node.node.text for node in response.source_nodes
        ]
        qca_sample.predicted_node_id = [
            node.node.node_id for node in response.source_nodes
        ]
        qca_sample.predicted_answer_by = self.created_by
        return qca_sample

    async def agenerate_predicted_qca_dataset(self, rag_qca_dataset):
        print("Starting to generate QCA dataset for [[predicted]].")
        tasks = []
        for qca_sample in rag_qca_dataset.examples:
            tasks.append(self.agenerate_predicted_qca_sample(qca_sample))
        predicted_examples = await run_jobs(tasks, self._show_progress, self._workers)
        predicted_qca_dataset = PaiRagQcaDataset(
            examples=predicted_examples, labelled=True, predicted=True
        )
        predicted_qca_dataset.save_json(self.qca_dataset_path)
        return predicted_qca_dataset
