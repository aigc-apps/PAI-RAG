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
from pai_rag.evaluation.base.rag_qca_dataset import RagQCADataset
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabelledRagDataset,
)
import json
import os
import logging

logger = logging.getLogger(__name__)


class RagQCAGenerator:
    def __init__(
        self, llm, vector_index: VectorStoreIndex = None, persist_path: str = None
    ):
        self._llm = llm
        self._vector_index = vector_index._vector_index
        self.question_gen_query = DEFAULT_QUESTION_GENERATION_QUERY.format(
            num_questions_per_chunk=3
        )
        self.text_question_template = PromptTemplate(DEFAULT_QUESTION_GENERATION_PROMPT)
        self.text_question_answer_template = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)
        model_name = self._llm.metadata.model_name
        self.created_by = CreatedBy(type=CreatedByType.AI, model_name=model_name)
        self.persist_path = persist_path
        self._show_progress = True
        self._workers = 2

    async def agenerate_qca_dataset(
        self,
    ) -> LabelledRagDataset:
        examples: List[RagQCADataset] = []
        docs = self._vector_index._docstore.docs
        nodes = list(docs.values())
        query_tasks = []
        for node in nodes:
            prompt_str = self.text_question_template.format(
                context_str=node.text, num_questions_per_chunk=3
            )
            task = self._llm.acomplete(prompt=prompt_str)
            query_tasks.append(task)
        responses = await run_jobs(query_tasks, self._show_progress, self._workers)
        for _, response in enumerate(responses):
            result = str(response).strip().split("\n")
            cleaned_questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            cleaned_questions = [
                question for question in cleaned_questions if len(question) > 0
            ]
            qr_tasks = []
            for query in cleaned_questions:
                # build summary index off of node (i.e. context)
                prompt_str = self.text_question_answer_template.format(
                    context_str=node.text, query_str=query
                )
                qr_task = self._llm.acomplete(prompt=prompt_str)
                qr_tasks.append(qr_task)
            answer_responses: List[RESPONSE_TYPE] = await run_jobs(
                qr_tasks, self._show_progress, self._workers
            )
            for (
                question,
                answer_response,
            ) in zip(cleaned_questions, answer_responses):
                example = RagQCADataset(
                    query=question,
                    reference_answer=str(answer_response),
                    reference_contexts=[node.text],
                    reference_node_id=[node.node_id],
                    reference_answer_by=self.created_by,
                    query_by=self.created_by,
                )
                examples.append(example)
        labelled_qca_dataset = LabelledRagDataset(examples=examples)
        labelled_qca_dataset_path = os.path.join(
            self.persist_path, "labelled_qca_dataset.json"
        )
        with open(labelled_qca_dataset_path, "w", encoding="utf-8") as f:
            json.dump(labelled_qca_dataset.dict(), f, indent=4, ensure_ascii=False)
        return labelled_qca_dataset
