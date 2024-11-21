from typing import Optional, Any
from llama_index.core.indices import VectorStoreIndex
from pai_rag.utils.prompt_template import (
    DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL,
)
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core.async_utils import run_jobs
from pai_rag.evaluation.dataset.rag_qca_dataset import PaiRagQcaDataset
from pai_rag.evaluation.generator.rag_qca_generator import RagQcaGenerator
import os
from loguru import logger
from pai_rag.integrations.query_engine.pai_retriever_query_engine import (
    PaiRetrieverQueryEngine,
)
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_TEXT_QA_PROMPT_SEL,
)
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.prompts import BasePromptTemplate
import asyncio


class MultimodalQcaGenerator(RagQcaGenerator):
    def __init__(
        self,
        labelled_llm,
        predicted_llm,
        vector_index: VectorStoreIndex = None,
        query_engine: PaiRetrieverQueryEngine = None,
        persist_path: str = None,
        qca_dataset_path: str = None,
        enable_multi_modal: bool = False,
        text_qa_template: Optional[BasePromptTemplate] = None,
        multimodal_qa_template: Optional[BasePromptTemplate] = None,
    ):
        super().__init__(
            labelled_llm, vector_index, query_engine, persist_path, enable_multi_modal
        )

        self.qca_dataset_path = qca_dataset_path or os.path.join(
            self.persist_path, "qca_dataset.json"
        )

        self._multimodal_qa_template = multimodal_qa_template or PromptTemplate(
            template=DEFAULT_MULTI_MODAL_IMAGE_QA_PROMPT_TMPL
        )
        self._text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
        self._multimodal_llm = predicted_llm or labelled_llm

    async def agenerate_qca_dataset(self, stage):
        rag_qca_dataset = self.load_qca_dataset()
        if rag_qca_dataset and rag_qca_dataset.labelled:
            if stage == "labelled":
                logger.info(
                    "Labelled [multi-modal] QCA dataset already exists. Skipping labelled stage."
                )
                return rag_qca_dataset.examples
            elif stage == "predicted":
                if rag_qca_dataset.predicted:
                    logger.info(
                        "Predicted [multi-modal] QCA dataset already exists. Skipping predicted stage."
                    )
                    return rag_qca_dataset.examples
                else:
                    return await self.agenerate_predicted_qca_dataset(rag_qca_dataset)
            else:
                raise ValueError(f"Invalid stage: {stage}")
        else:
            return await self.agenerate_labelled_qca_dataset()

    async def agenerate_predicted_qca_dataset(self, rag_qca_dataset):
        logger.info("Starting to generate [multi-modal] QCA dataset for [[predicted]].")
        tasks = []
        for qca_sample in rag_qca_dataset.examples:
            if self.enable_multi_modal:
                tasks.append(self.agenerate_predicted_multimodal_qca_sample(qca_sample))
            else:
                tasks.append(self.agenerate_predicted_qca_sample(qca_sample))
        predicted_examples = await run_jobs(tasks, self._show_progress, self._workers)
        predicted_qca_dataset = PaiRagQcaDataset(
            examples=predicted_examples, labelled=True, predicted=True
        )
        predicted_qca_dataset.save_json(self.qca_dataset_path)
        return predicted_qca_dataset

    async def agenerate_predicted_multimodal_qca_sample(
        self, qca_sample, **response_kwargs: Any
    ):
        image_url_list = qca_sample.reference_image_url_list
        image_documents = load_image_urls(image_url_list)
        image_context_str = "\n\n".join(image_url_list)
        reference_contexts = qca_sample.reference_contexts
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
        response = llm_response.text or "Empty Response"

        await asyncio.sleep(3)

        qca_sample.predicted_answer = response
        return qca_sample

    async def agenerate_predicted_qca_sample(self, qca_sample, **response_kwargs: Any):
        reference_contexts = qca_sample.reference_contexts
        text_context_str = "\n\n".join(reference_contexts)
        query_str = qca_sample.query
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        response: RESPONSE_TEXT_TYPE
        response = await self._llm.apredict(
            text_qa_template,
            context_str=text_context_str,
            **response_kwargs,
        )
        await asyncio.sleep(3)

        qca_sample.predicted_answer = response
        return qca_sample
