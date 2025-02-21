import os
import re
import json
from loguru import logger
from typing import List, Optional, Any
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.async_utils import run_jobs
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import TextNode, ImageNode
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from pai_rag.evaluation.dataset.rag_qca_dataset_refactor import (
    QcaSample,
    QcapSample,
    QcaDataset,
    QcapDataset,
)
from pai_rag.evaluation.dataset.state_manager import StateManager, DatasetState
from pai_rag.app.api.models import PaiQueryBundle
from pai_rag.integrations.query_engine.pai_retriever_query_engine import (
    PaiRetrieverQueryEngine,
)
from pai_rag.evaluation.utils.file_utils import list_files_in_directory
from pai_rag.utils.prompt_template import (
    DEFAULT_QUESTION_GENERATION_PROMPT,
    DEFAULT_MULTI_MODAL_QUESTION_GENERATION_PROMPT,
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL,
)
from pai_rag.integrations.synthesizer.pai_synthesizer import DEFAULT_EMPTY_RESPONSE_GEN
from llama_index.core.prompts import BasePromptTemplate
import asyncio


class RagQcaGenerator:
    def __init__(
        self,
        llm: None = None,
        labelled_llm: None = None,
        predicted_llm: None = None,
        vector_index: VectorStoreIndex = None,
        query_engine: PaiRetrieverQueryEngine = None,
        persist_path: str = None,
        enable_multi_modal: bool = False,
        multimodal_qa_template: Optional[BasePromptTemplate] = None,
        state_manager: StateManager = None,
    ):
        self._llm = llm
        self._vector_index = vector_index._vector_index
        self._query_engine = query_engine
        self.text_question_template = PromptTemplate(DEFAULT_QUESTION_GENERATION_PROMPT)
        self.multi_modal_question_template = PromptTemplate(
            DEFAULT_MULTI_MODAL_QUESTION_GENERATION_PROMPT
        )
        self.text_question_answer_template = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)
        self.multi_modal_question_answer_template = PromptTemplate(
            DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL
        )
        self.qca_dataset_path = os.path.join(persist_path, "qca_dataset.jsonl")
        self.qcap_dataset_path = os.path.join(persist_path, "qcap_dataset.jsonl")
        self.state_manager = state_manager or StateManager(
            os.path.join(persist_path, "state.json")
        )
        self._show_progress = True
        self._workers = 2
        self.enable_multi_modal = enable_multi_modal
        self._multimodal_qa_template = multimodal_qa_template or PromptTemplate(
            template=DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL
        )
        self._multimodal_llm = predicted_llm or labelled_llm

    async def agenerate_all_dataset(self, dataset=None, dataset_path=None):
        if self.state_manager.is_completed(DatasetState.QCAP):
            logger.info(
                "Predicted QCA dataset already exists. Skipping predicted stage."
            )
            rag_qcap_dataset = QcapDataset.from_json(self.qcap_dataset_path)
            return rag_qcap_dataset.samples
        if dataset == "crag":
            _ = await self.agenerate_labelled_qca_dataset_for_crag(dataset_path)
        else:
            _ = await self.agenerate_labelled_qca_dataset()

        return await self.agenerate_predicted_qca_dataset()

    async def agenerate_labelled_multimodal_qca_sample(self, node):
        assert isinstance(
            self._multimodal_llm, OpenAIMultiModal
        ), "Multi-modal LLM must be provided to understand image documents."
        image_url_infos = node.metadata.get("image_info_list", None)
        if image_url_infos:
            image_url_list = [
                image_url_info.get("image_url", None)
                for image_url_info in image_url_infos
            ]
            image_context_str = "\n\n".join(image_url_list)
            image_documents = load_image_urls(image_url_list)

        else:
            image_url_list = []
            image_context_str = ""
            image_documents = None

        context_str = f"{node.text}\n\n图片链接列表: \n\n{image_context_str}\n\n"
        prompt_str = self.multi_modal_question_template.format(
            context_str=context_str, num_questions_per_chunk=1
        )
        response = await self._multimodal_llm.acomplete(
            prompt=prompt_str, image_documents=image_documents
        )
        result = str(response).strip().split("\n")
        cleaned_questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        cleaned_questions = [
            question for question in cleaned_questions if len(question) > 0
        ]
        qr_tasks = []
        for query in cleaned_questions:
            prompt_str = self.multi_modal_question_answer_template.format(
                context_str=context_str, query_str=query
            )
            qr_task = self._multimodal_llm.acomplete(
                prompt=prompt_str, image_documents=image_documents
            )
            qr_tasks.append(qr_task)
        answer_responses: List[RESPONSE_TYPE] = await run_jobs(
            qr_tasks, self._show_progress, self._workers
        )
        for (
            question,
            answer_response,
        ) in zip(cleaned_questions, answer_responses):
            sample = QcaSample(
                query={
                    "query_text": question,
                    "source": {
                        "name": "AI",
                        "model": self._multimodal_llm.metadata.model_name,
                    },
                },
                contexts=[
                    {
                        "type": "TextNode",
                        "text": node.text,
                        "node_id": node.node_id,
                        "metadata": {"image_url_list": image_url_list},
                    }
                ],
                answer={
                    "answer_text": str(answer_response),
                    "source": {
                        "name": "AI",
                        "model": self._multimodal_llm.metadata.model_name,
                    },
                },
            )
        return sample

    async def agenerate_labelled_qca_sample(self, node):
        prompt_str = self.text_question_template.format(
            context_str=node.text, num_questions_per_chunk=1
        )
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
            if isinstance(self._llm, OpenAIMultiModal):
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
            sample = QcaSample(
                query={
                    "query_text": question,
                    "source": {"name": "AI", "model": self._llm.metadata.model_name},
                },
                contexts=[
                    {
                        "type": "TextNode",
                        "text": node.text,
                        "node_id": node.node_id,
                        "metadata": {"image_url_list": []},
                    }
                ],
                answer={
                    "answer_text": str(answer_response),
                    "source": {"name": "AI", "model": self._llm.metadata.model_name},
                },
            )
            return sample

    async def agenerate_labelled_qca_dataset(
        self,
    ):
        if self.state_manager.is_completed(DatasetState.QCA):
            logger.info("Labelled QCA dataset already exists. Skipping labelled stage.")
            rag_qca_dataset = QcaDataset.from_json(self.qca_dataset_path)
            return rag_qca_dataset.samples
        else:
            logger.info("Starting to generate QCA dataset for [[labelled]].")
            docs = self._vector_index._docstore.docs
            nodes = list(docs.values())
            tasks = []
            for node in nodes:
                if self.enable_multi_modal:
                    if type(node) is TextNode:
                        tasks.append(
                            self.agenerate_labelled_multimodal_qca_sample(node)
                        )
                else:
                    tasks.append(self.agenerate_labelled_qca_sample(node))
            samples = await run_jobs(tasks, self._show_progress, self._workers)
            labelled_qca_dataset = QcaDataset(samples=samples)
            labelled_qca_dataset.save_json(self.qca_dataset_path)
            self.state_manager.mark_completed(DatasetState.QCA)
            return labelled_qca_dataset.samples

    async def agenerate_predicted_multimodal_qca_sample(self, qca_sample):
        query_bundle = PaiQueryBundle(query_str=qca_sample.query.query_text)
        response = await self._query_engine.aquery(query_bundle)
        await asyncio.sleep(3)
        predicted_contexts = []
        for node in response.source_nodes:
            if type(node.node) is TextNode:
                image_url_infos = node.node.metadata.get("image_info_list", None)
                image_url_list = []
                if image_url_infos:
                    image_url_list = [
                        image_url_info.get("image_url", None)
                        for image_url_info in image_url_infos
                    ]
                predicted_contexts.append(
                    {
                        "type": "TextNode",
                        "text": node.node.text,
                        "node_id": node.node.node_id,
                        "metadata": {
                            "node_score": node.score,
                            "image_url_list": image_url_list,
                        },
                    }
                )
            elif type(node.node) is ImageNode:
                predicted_contexts.append(
                    {
                        "type": "ImageNode",
                        "node_id": node.node.node_id,
                        "metadata": {
                            "node_score": node.score,
                            "image_url": node.node.image_url,
                        },
                    }
                )
        qcap_sample = QcapSample(
            qca=qca_sample,
            prediction={
                "contexts": predicted_contexts,
                "answer": {
                    "answer_text": str(response.response),
                    "source": {
                        "name": "AI",
                        "model": self._multimodal_llm.metadata.model_name,
                    },
                },
            },
            mode="multimodal",
        )
        return qcap_sample

    async def agenerate_predicted_qca_sample(self, qca_sample):
        query_bundle = PaiQueryBundle(query_str=qca_sample.query.query_text)
        response = await self._query_engine.aquery(query_bundle)
        await asyncio.sleep(3)

        qcap_sample = QcapSample(
            qca=qca_sample,
            prediction={
                "contexts": [
                    {
                        "type": "TextNode",
                        "text": node.node.text,
                        "node_id": node.node.node_id,
                        "metadata": {"node_score": node.score, "image_url_list": []},
                    }
                    for node in response.source_nodes
                ],
                "answer": {
                    "answer_text": str(response.response),
                    "source": {"name": "AI", "model": self._llm.metadata.model_name},
                },
            },
            mode="text",
        )
        return qcap_sample

    async def agenerate_predicted_qca_dataset(self):
        if not self.state_manager.is_completed(DatasetState.QCA):
            logger.info(
                "Error: No existing QCA dataset found. You need first create or provided a QcaDataset file."
            )
            return
        if self.state_manager.is_completed(DatasetState.QCAP):
            logger.info(
                "Predicted QCA dataset already exists. Skipping predicted stage."
            )
            rag_qcap_dataset = QcapDataset.from_json(self.qcap_dataset_path)
            return rag_qcap_dataset.samples
        else:
            logger.info("Starting to generate QCAP dataset for [[predicted]].")
            tasks = []
            rag_qca_dataset = QcaDataset.from_json(self.qca_dataset_path)
            for qca_sample in rag_qca_dataset.samples:
                if self.enable_multi_modal:
                    tasks.append(
                        self.agenerate_predicted_multimodal_qca_sample(qca_sample)
                    )
                else:
                    tasks.append(self.agenerate_predicted_qca_sample(qca_sample))
            predicted_samples = await run_jobs(
                tasks, self._show_progress, self._workers
            )
            predicted_qca_dataset = QcapDataset(samples=predicted_samples)
            predicted_qca_dataset.save_json(self.qcap_dataset_path)
            self.state_manager.mark_completed(DatasetState.QCAP)
            return predicted_qca_dataset.samples

    ##### for crag #####
    async def agenerate_labelled_qca_dataset_for_crag(self, dataset_path: str = None):
        if self.state_manager.is_completed(DatasetState.QCA):
            logger.info("Labelled QCA dataset already exists. Skipping labelled stage.")
            rag_qca_dataset = QcaDataset.from_json(self.qca_dataset_path)
            return rag_qca_dataset.samples
        else:
            logger.info("Starting to generate QCA dataset for [[labelled]].")
            data_files = list_files_in_directory(dataset_path)
            samples = []
            for data_file in data_files:
                with open(data_file, "r", encoding="utf-8") as file:
                    json_lines = [line.strip() for line in file.readlines()]
                    for data in json_lines:
                        json_data = json.loads(data)
                        sample = QcaSample(
                            query={
                                "query_text": json_data["query"],
                            },
                            contexts=[
                                {
                                    "type": "TextNode",
                                    "text": si["page_snippet"],
                                    "node_id": f"{json_data['interaction_id']}__{i}",
                                    "metadata": {"image_url_list": []},
                                }
                                for i, si in enumerate(json_data["search_results"])
                            ],
                            answer={
                                "answer_text": str(json_data["answer"]),
                            },
                        )
                        samples.append(sample)
            labelled_qca_dataset = QcaDataset(samples=samples)
            labelled_qca_dataset.save_json(self.qca_dataset_path)
            self.state_manager.mark_completed(DatasetState.QCA)
            return labelled_qca_dataset.samples

    #### for multimodal test data####
    async def agenerate_predicted_multimodal_dataset_only_via_vlm(self):
        if not self.state_manager.is_completed(DatasetState.QCA):
            logger.info(
                "Error: No existing QCA dataset found. You need first create or provided a QcaDataset file."
            )
            return
        if self.state_manager.is_completed(DatasetState.QCAP):
            logger.info(
                "Predicted QCA dataset already exists. Skipping predicted stage."
            )
            rag_qcap_dataset = QcapDataset.from_json(self.qcap_dataset_path)
            return rag_qcap_dataset.samples
        else:
            logger.info(
                "Starting to generate multimodal QCAP dataset for [[predicted]] only via vlm."
            )
            tasks = []
            rag_qca_dataset = QcaDataset.from_json(self.qca_dataset_path)
            for qca_sample in rag_qca_dataset.samples:
                tasks.append(
                    self.agenerate_predicted_multimodal_qca_sample_only_via_vlm(
                        qca_sample
                    )
                )
            predicted_samples = await run_jobs(
                tasks, self._show_progress, self._workers
            )
            predicted_qca_dataset = QcapDataset(samples=predicted_samples)
            predicted_qca_dataset.save_json(self.qcap_dataset_path)
            self.state_manager.mark_completed(DatasetState.QCAP)
            return predicted_qca_dataset.samples

    async def agenerate_predicted_multimodal_qca_sample_only_via_vlm(
        self, qca_sample, **response_kwargs: Any
    ):
        image_url_list = qca_sample.get_reference_image_url_list()
        reference_contexts = qca_sample.get_reference_node_texts()

        image_documents = load_image_urls(image_url_list)
        image_context_str = "\n\n".join(image_url_list)
        text_context_str = "\n\n".join(reference_contexts)
        query_str = qca_sample.query
        context_str = f"{text_context_str}\n\n图片链接列表: \n\n{image_context_str}\n\n"
        fmt_prompt = self._multimodal_qa_template.format(
            context_str=context_str, query_str=query_str
        )
        llm_response = self._multimodal_llm.complete(
            prompt=fmt_prompt,
            image_documents=image_documents,
            **response_kwargs,
        )
        response = llm_response.text or DEFAULT_EMPTY_RESPONSE_GEN
        await asyncio.sleep(3)

        qcap_sample = QcapSample(
            qca=qca_sample,
            prediction={
                "contexts": [],
                "answer": {
                    "answer_text": str(response),
                    "source": {
                        "name": "AI",
                        "model": self._multimodal_llm.metadata.model_name,
                    },
                },
            },
            mode="multimodal",
        )
        return qcap_sample
