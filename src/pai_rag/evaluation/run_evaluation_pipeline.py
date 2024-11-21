import os
import asyncio
from pathlib import Path
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.rag_module import (
    resolve,
    resolve_data_loader,
    resolve_vector_index,
    resolve_query_engine,
)
from pai_rag.evaluation.generator.rag_qca_generator import RagQcaGenerator
from pai_rag.evaluation.generator.multimodal_qca_generator import MultimodalQcaGenerator
from pai_rag.integrations.llms.pai.pai_multi_modal_llm import (
    PaiMultiModalLlm,
)
from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.evaluation.evaluator.base_evaluator import BaseEvaluator
from loguru import logger

from pai_rag.integrations.llms.pai.llm_config import parse_llm_config
from pai_rag.integrations.llms.pai.llm_utils import create_llm, create_multi_modal_llm


_BASE_DIR = Path(__file__).parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(
    _BASE_DIR, "evaluation/settings_eval_for_text.toml"
)


def _create_components(
    config_file, exp_name, eval_model_source, eval_model_name
) -> None:
    """Create all components from the default config file."""
    config = RagConfigManager.from_file(config_file).get_value()
    mode = "image" if config.retriever.search_image else "text"
    config.synthesizer.use_multimodal_llm = True if mode == "image" else False

    logger.info(f"Creating RAG evaluation components for mode: {mode}...")

    config.index.vector_store.persist_path = (
        f"{config.index.vector_store.persist_path}__{exp_name}"
    )
    data_loader = resolve_data_loader(config)
    vector_index = resolve_vector_index(config)
    query_engine = resolve_query_engine(config)
    eval_llm_config_data = {
        "source": eval_model_source.lower(),
        "model": eval_model_name,
        "max_tokens": 1024,
    }
    eval_llm_config = parse_llm_config(eval_llm_config_data)
    if mode == "text":
        llm = resolve(cls=PaiLlm, llm_config=config.llm)
        eval_llm = create_llm(eval_llm_config)
    else:
        llm = resolve(cls=PaiMultiModalLlm, llm_config=config.multimodal_llm)
        eval_llm = create_multi_modal_llm(eval_llm_config)

    qca_generator = RagQcaGenerator(
        llm=llm,
        vector_index=vector_index,
        query_engine=query_engine,
        persist_path=config.index.vector_store.persist_path,
        enable_multi_modal=True if mode == "image" else False,
    )

    evaluator = BaseEvaluator(
        llm=eval_llm,
        persist_path=config.index.vector_store.persist_path,
        enable_multi_modal=True if mode == "image" else False,
    )

    return data_loader, qca_generator, evaluator


def _create_multimodal_components(
    config_file,
    exp_name,
    eval_model_source,
    eval_model_name,
    tested_multimodal_llm_config,
    qca_dataset_path: str = None,
) -> None:
    """Create all components from the default config file."""
    config = RagConfigManager.from_file(config_file).get_value()
    mode = "image" if config.retriever.search_image else "text"
    config.synthesizer.use_multimodal_llm = True if mode == "image" else False

    logger.info(f"Creating RAG evaluation components for mode: {mode}...")

    config.index.vector_store.persist_path = (
        f"{config.index.vector_store.persist_path}__{exp_name}"
    )
    data_loader = resolve_data_loader(config)
    vector_index = resolve_vector_index(config)
    query_engine = resolve_query_engine(config)
    eval_llm_config_data = {
        "source": eval_model_source.lower(),
        "model": eval_model_name,
        "max_tokens": 1024,
    }
    eval_llm_config = parse_llm_config(eval_llm_config_data)
    eval_llm = create_multi_modal_llm(eval_llm_config)

    llm = resolve(cls=PaiMultiModalLlm, llm_config=config.multimodal_llm)
    tested_multimodal_llm_config = parse_llm_config(tested_multimodal_llm_config)
    tested_multimodal_llm = create_multi_modal_llm(tested_multimodal_llm_config)

    multimodal_qca_generator = MultimodalQcaGenerator(
        labelled_llm=llm,
        predicted_llm=tested_multimodal_llm,
        vector_index=vector_index,
        query_engine=query_engine,
        persist_path=config.index.vector_store.persist_path,
        qca_dataset_path=qca_dataset_path,
        enable_multi_modal=True if mode == "image" else False,
    )
    if qca_dataset_path:
        persist_path = os.path.join("localdata/eval_exp_data", exp_name)
        os.makedirs(persist_path, exist_ok=True)
    else:
        persist_path = config.index.vector_store.persist_path
    evaluator = BaseEvaluator(
        llm=eval_llm,
        persist_path=persist_path,
        qca_dataset_path=qca_dataset_path,
        enable_multi_modal=True if mode == "image" else False,
    )

    return data_loader, multimodal_qca_generator, evaluator


def run_evaluation_pipeline(
    config=None,
    oss_path=None,
    data_path=None,
    pattern=None,
    exp_name="default",
    eval_model_source=None,
    eval_model_name=None,
):
    assert (oss_path is not None) or (
        data_path is not None
    ), "Must provide either local path or oss path."
    assert (oss_path is None) or (
        data_path is None
    ), f"Can not provide both local path '{data_path}' and oss path '{oss_path}'."

    data_loader, qca_generator, evaluator = _create_components(
        config, exp_name, eval_model_source, eval_model_name
    )
    data_loader.load_data(
        file_path_or_directory=data_path,
        filter_pattern=pattern,
        oss_path=oss_path,
        from_oss=oss_path is not None,
        enable_raptor=False,
    )

    _ = asyncio.run(qca_generator.agenerate_qca_dataset(stage="labelled"))
    _ = asyncio.run(qca_generator.agenerate_qca_dataset(stage="predicted"))
    retrieval_result = asyncio.run(evaluator.aevaluation(stage="retrieval"))
    response_result = asyncio.run(evaluator.aevaluation(stage="response"))
    logger.info(
        "retrieval_result", retrieval_result, "response_result", response_result
    )
    return {"retrieval": retrieval_result, "response": response_result}


def run_multimodal_evaluation_pipeline(
    config=None,
    oss_path=None,
    qca_dataset_path=None,
    data_path=None,
    pattern=None,
    exp_name="default",
    eval_model_source=None,
    eval_model_name=None,
    tested_multimodal_llm_config=None,
):
    data_loader, multimodal_qca_generator, evaluator = _create_multimodal_components(
        config,
        exp_name,
        eval_model_source,
        eval_model_name,
        tested_multimodal_llm_config,
        qca_dataset_path,
    )
    if qca_dataset_path:
        _ = asyncio.run(
            multimodal_qca_generator.agenerate_qca_dataset(stage="predicted")
        )
        response_result = asyncio.run(evaluator.aevaluation(stage="response"))
        return {"response": response_result}

    assert (oss_path is not None) or (
        data_path is not None
    ), "Must provide either local path or oss path."
    assert (oss_path is None) or (
        data_path is None
    ), f"Can not provide both local path '{data_path}' and oss path '{oss_path}'."

    data_loader.load_data(
        file_path_or_directory=data_path,
        filter_pattern=pattern,
        oss_path=oss_path,
        from_oss=oss_path is not None,
        enable_raptor=False,
    )

    _ = asyncio.run(multimodal_qca_generator.agenerate_qca_dataset(stage="labelled"))
    _ = asyncio.run(multimodal_qca_generator.agenerate_qca_dataset(stage="predicted"))
    response_result = asyncio.run(evaluator.aevaluation(stage="response"))
    return {"response": response_result}
