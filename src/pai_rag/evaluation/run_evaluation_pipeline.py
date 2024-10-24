import os
import asyncio
from pathlib import Path
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.rag_data_loader import RagDataLoader
from pai_rag.core.rag_module import (
    resolve,
    resolve_data_loader,
    resolve_llm,
    resolve_vector_index,
    resolve_query_engine,
)
from pai_rag.evaluation.generator.rag_qca_generator import RagQcaGenerator
from pai_rag.integrations.llms.pai.pai_multi_modal_llm import (
    PaiMultiModalLlm,
)
from pai_rag.evaluation.evaluator.base_evaluator import BaseEvaluator
import logging


logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(
    _BASE_DIR, "evaluation/settings_eval.toml"
)


def _create_data_loader(
    config_file, name, enable_raptor: bool = False
) -> RagDataLoader:
    config = RagConfigManager.from_file(config_file).get_value()

    config.index.vector_store.persist_path = (
        f"{config.index.vector_store.persist_path}__{name}"
    )
    data_loader = resolve_data_loader(config)
    vector_index = resolve_vector_index(config)
    query_engine = resolve_query_engine(config)

    return data_loader, vector_index, query_engine


def _create_qca_generator(config_file, name, vector_index, query_engine):
    config = RagConfigManager.from_file(config_file).get_value()
    multimodal_llm = resolve(cls=PaiMultiModalLlm, llm_config=config.multimodal_llm)
    persist_path = f"{config.index.vector_store.persist_path}__{name}"
    qca_generator = RagQcaGenerator(
        llm=multimodal_llm,
        vector_index=vector_index,
        query_engine=query_engine,
        persist_path=persist_path,
    )
    return qca_generator


def _create_base_evaluator(config_file, name):
    config = RagConfigManager.from_file(config_file).get_value()
    llm = resolve_llm(config)
    persist_path = f"{config.index.vector_store.persist_path}__{name}"
    return BaseEvaluator(
        llm=llm,
        persist_path=persist_path,
    )


def run_evaluation_pipeline(
    config=None,
    oss_path=None,
    data_path=None,
    pattern=None,
    enable_raptor=False,
    name="default",
):
    assert (oss_path is not None) or (
        data_path is not None
    ), "Must provide either local path or oss path."
    assert (oss_path is None) or (
        data_path is None
    ), f"Can not provide both local path '{data_path}' and oss path '{oss_path}'."

    data_loader, vector_index, query_engine = _create_data_loader(
        config, name, enable_raptor
    )
    data_loader.load_data(
        file_path_or_directory=data_path,
        filter_pattern=pattern,
        oss_path=oss_path,
        from_oss=oss_path is not None,
        enable_raptor=enable_raptor,
    )
    qca_generator = _create_qca_generator(config, name, vector_index, query_engine)
    _ = asyncio.run(qca_generator.agenerate_qca_dataset(stage="labelled"))
    _ = asyncio.run(qca_generator.agenerate_qca_dataset(stage="predicted"))
    evaluator = _create_base_evaluator(config, name)
    retrieval_result = asyncio.run(evaluator.aevaluation(stage="retrieval"))
    response_result = asyncio.run(evaluator.aevaluation(stage="response"))
    print("retrieval_result", retrieval_result, "response_result", response_result)
    return {"retrieval": retrieval_result, "response": response_result}
