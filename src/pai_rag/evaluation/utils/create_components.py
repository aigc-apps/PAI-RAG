import os
from loguru import logger
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.rag_module import (
    resolve,
    resolve_data_loader,
    resolve_vector_index,
    resolve_query_engine,
)
from pai_rag.integrations.llms.pai.llm_config import parse_llm_config
from pai_rag.integrations.llms.pai.llm_utils import create_llm, create_multi_modal_llm
from pai_rag.evaluation.generator.rag_qca_generator import RagQcaGenerator
from pai_rag.integrations.llms.pai.pai_multi_modal_llm import (
    PaiMultiModalLlm,
)
from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.evaluation.evaluator.base_evaluator import BaseEvaluator
from pai_rag.evaluation.evaluator.pai_evaluator import PaiEvaluator
from pai_rag.evaluation.dataset.crag.crag_jsonl_reader import CragJsonLReader
from pai_rag.evaluation.dataset.crag.crag_data_loader import CragDataLoader
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding


def get_rag_config_and_mode(config_file, exp_name):
    """Create all components from the default config file."""
    config = RagConfigManager.from_file(config_file).get_value()
    mode = "image" if config.retriever.search_image else "text"
    config.synthesizer.use_multimodal_llm = True if mode == "image" else False

    logger.info(f"Creating RAG evaluation configuration for mode: {mode}...")
    config.index.vector_store.persist_path = (
        f"{config.index.vector_store.persist_path}__{exp_name}"
    )
    exist_flag = os.path.exists(config.index.vector_store.persist_path)
    return config, mode, exist_flag


def get_rag_components(config, dataset=None):
    vector_index = resolve_vector_index(config)
    query_engine = resolve_query_engine(config)

    if dataset is not None and dataset == "crag":
        directory_reader = resolve(
            cls=PaiDataReader,
            reader_config=config.data_reader,
        )
        directory_reader.file_readers[".jsonl"] = CragJsonLReader()
        embed_model = resolve(cls=PaiEmbedding, embed_config=config.embedding)
        data_loader = CragDataLoader(
            data_reader=directory_reader,
            embed_model=embed_model,
            vector_index=vector_index,
        )
    else:
        data_loader = resolve_data_loader(config)

    return data_loader, vector_index, query_engine


def get_eval_components(
    config,
    vector_index,
    query_engine,
    mode,
    eval_model_source,
    eval_model_name,
    use_pai_eval=False,
):
    if mode == "text":
        llm = resolve(cls=PaiLlm, llm_config=config.llm)
    else:
        llm = resolve(cls=PaiMultiModalLlm, llm_config=config.multimodal_llm)

    qca_generator = RagQcaGenerator(
        llm=llm,
        vector_index=vector_index,
        query_engine=query_engine,
        persist_path=config.index.vector_store.persist_path,
        enable_multi_modal=True if mode == "image" else False,
    )

    if use_pai_eval:
        model_config = {
            "model_name": eval_model_name,
            "is_self_host": False,
            "use_function_call": True,
        }
        evaluator = PaiEvaluator(
            llm_config=model_config,
            persist_path=config.index.vector_store.persist_path,
        )
    else:
        eval_llm_config_data = {
            "source": eval_model_source.lower(),
            "model": eval_model_name,
            "max_tokens": 1024,
        }
        eval_llm_config = parse_llm_config(eval_llm_config_data)
        if mode == "text":
            eval_llm = create_llm(eval_llm_config)
        else:
            eval_llm = create_multi_modal_llm(eval_llm_config)
        evaluator = BaseEvaluator(
            llm=eval_llm,
            persist_path=config.index.vector_store.persist_path,
            enable_multi_modal=True if mode == "image" else False,
            use_granular_metrics=True,
        )
    return qca_generator, evaluator
