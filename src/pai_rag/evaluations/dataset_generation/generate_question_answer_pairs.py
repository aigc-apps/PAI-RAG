import os
import json
import logging
from llama_index.core.llama_dataset import LabelledRagDataset
from pai_rag.evaluations.dataset_generation.generate_dataset import (
    GenerateDatasetPipeline,
)

logger = logging.getLogger(__name__)


async def customized_generate_qas(path):
    pipeline = GenerateDatasetPipeline()
    qas = await pipeline.agenerate_dataset()
    pipeline.save_json(qas, path)


async def load_question_answer_pairs(path):
    if not os.path.exists(path):
        logger.info("[Response Evaluation] qa_dataset not exists, generating... ")
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        await customized_generate_qas(path)
    logger.info(
        f"[Response Evaluation] loading generated qa_dataset from path {path}. "
    )
    qa_dataset = LabelledRagDataset.from_json(path)
    return qa_dataset


async def load_question_answer_pairs_json(path, overwrite):
    file_exists = os.path.exists(path)
    if file_exists and not overwrite:
        logger.info(
            f"[Evaluation] qa_dataset '{path}' already exists, do not need to regenerate and overwrite."
        )
    else:
        logger.info(
            f"[Evaluation] qa_dataset '{path}' (re)generating and overwriting..."
        )
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        await customized_generate_qas(path)
    logger.info(f"[Evaluation] loading generated qa_dataset from path {path}. ")
    with open(path) as f:
        qa_dataset = json.load(f)
    return qa_dataset
