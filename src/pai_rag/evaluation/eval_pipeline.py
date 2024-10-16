import os
import asyncio
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.core.rag_data_loader import RagDataLoader
from pai_rag.integrations.embeddings.pai.pai_embedding import PaiEmbedding
from pai_rag.integrations.embeddings.pai.pai_embedding_config import parse_embed_config
from pai_rag.integrations.embeddings.pai.pai_multimodal_embedding import (
    PaiMultiModalEmbedding,
)
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex
from pai_rag.integrations.index.pai.vector_store_config import PaiVectorIndexConfig
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import (
    NodeParserConfig,
    PaiNodeParser,
)
from pai_rag.integrations.nodes.raptor_nodes_enhance import RaptorProcessor
from pai_rag.integrations.readers.pai.pai_data_reader import (
    BaseDataReaderConfig,
    PaiDataReader,
)
from pai_rag.utils.oss_client import OssClient
from pai_rag.evaluation.generator.labelled_qca_generator import LabelledRagQcaGenerator
from pai_rag.evaluation.generator.predicted_qca_generator import (
    PredictedRagQcaGenerator,
)
from pai_rag.integrations.llms.pai.pai_llm import PaiLlm
from pai_rag.integrations.llms.pai.pai_multi_modal_llm import (
    PaiMultiModalLlm,
)
from pai_rag.integrations.llms.pai.llm_config import parse_llm_config
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
    config = RagConfiguration.from_file(config_file).get_value()
    config.rag.index.vector_store.persist_path = (
        f"{config.rag.index.vector_store.persist_path}__{name}"
    )

    oss_store = None
    if config.rag.oss_store.bucket:
        oss_store = OssClient(
            bucket_name=config.rag.oss_store.bucket,
            endpoint=config.rag.oss_store.endpoint,
        )

    data_reader_config = BaseDataReaderConfig.model_validate(config.rag.data_reader)
    data_reader = PaiDataReader(reader_config=data_reader_config, oss_store=oss_store)

    node_parser_config = NodeParserConfig.model_validate(config.rag.node_parser)
    node_parser = PaiNodeParser(parser_config=node_parser_config)

    embed_config = parse_embed_config(config.rag.embedding)
    embed_model = PaiEmbedding(embed_config)

    multi_modal_embed_config = parse_embed_config(config.rag.embedding.multi_modal)
    multi_modal_embed_model = PaiMultiModalEmbedding(multi_modal_embed_config)

    index_config = PaiVectorIndexConfig.model_validate(config.rag.index)
    vector_index = PaiVectorStoreIndex(
        vector_store_config=index_config.vector_store,
        enable_multimodal=index_config.enable_multimodal,
        embed_model=embed_model,
        multi_modal_embed_model=multi_modal_embed_model,
        enable_local_keyword_index=True,
    )

    raptor_processor = None
    if enable_raptor:
        raptor_processor = RaptorProcessor(
            tree_depth=config.rag.node_enhancement.tree_depth,
            max_clusters=config.rag.node_enhancement.max_clusters,
            threshold=config.rag.node_enhancement.threshold,
            embed_model=embed_model,
        )

    data_loader = RagDataLoader(
        data_reader=data_reader,
        node_parser=node_parser,
        raptor_processor=raptor_processor,
        embed_model=embed_model,
        multi_modal_embed_modal=multi_modal_embed_model,
        vector_index=vector_index,
    )

    return data_loader, vector_index


def _create_labelled_qca_generator(config_file, name, vector_index) -> None:
    config = RagConfiguration.from_file(config_file).get_value()
    llm_config = parse_llm_config(config.rag.llm)
    llm = PaiLlm(llm_config)
    persist_path = f"{config.rag.index.vector_store.persist_path}__{name}"
    qca_generator = LabelledRagQcaGenerator(
        llm=llm, vector_index=vector_index, persist_path=persist_path
    )
    return qca_generator


def _create_predicted_qca_generator(config_file, name, vector_index) -> None:
    config = RagConfiguration.from_file(config_file).get_value()
    llm_config = parse_llm_config(config.rag.llm)
    multimodal_llm = PaiMultiModalLlm(llm_config)
    persist_path = f"{config.rag.index.vector_store.persist_path}__{name}"
    predicted_qca_generator = PredictedRagQcaGenerator(
        llm=multimodal_llm,
        vector_index=vector_index,
        persist_path=persist_path,
    )
    return predicted_qca_generator


def _create_base_evaluator(config_file, name):
    config = RagConfiguration.from_file(config_file).get_value()
    llm_config = parse_llm_config(config.rag.llm)
    llm = PaiLlm(llm_config)
    persist_path = f"{config.rag.index.vector_store.persist_path}__{name}"
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

    data_loader, vector_index = _create_data_loader(config, name, enable_raptor)
    data_loader.load_data(
        file_path_or_directory=data_path,
        filter_pattern=pattern,
        oss_path=oss_path,
        from_oss=oss_path is not None,
        enable_raptor=enable_raptor,
    )
    qca_generator = _create_labelled_qca_generator(config, name, vector_index)
    asyncio.run(qca_generator.agenerate_labelled_qca_dataset())

    predicted_qca_generator = _create_predicted_qca_generator(
        config, name, vector_index
    )
    asyncio.run(predicted_qca_generator.agenerate_predicted_qca_dataset())
    evaluator = _create_base_evaluator(config, name)
    qcas = evaluator.load_predicted_qca_dataset()
    retrieval_result = asyncio.run(evaluator.aevaluation_for_retrieval(qcas))
    response_result = asyncio.run(evaluator.aevaluation_for_response(qcas))
    print("retrieval_result", retrieval_result, "response_result", response_result)
    return retrieval_result, response_result


# def exp_run(
#     config=None,
#     oss_path=None,
#     data_path=None,
#     pattern=None,
#     enable_raptor=False,
#     name=None,
# ):
#     assert (oss_path is not None) or (
#         data_path is not None
#     ), "Must provide either local path or oss path."
#     assert (oss_path is None) or (
#         data_path is None
#     ), f"Can not provide both local path '{data_path}' and oss path '{oss_path}'."

#     data_loader, vector_index = _create_data_loader(config, name, enable_raptor)
#     data_loader.load_data(
#         file_path_or_directory=data_path,
#         filter_pattern=pattern,
#         oss_path=oss_path,
#         from_oss=oss_path is not None,
#         enable_raptor=enable_raptor,
#     )
#     qca_generator = _create_labelled_qca_generator(config, name, vector_index)
#     asyncio.run(qca_generator.agenerate_labelled_qca_dataset())

#     predicted_qca_generator = _create_predicted_qca_generator(
#         config, name, vector_index
#     )
#     asyncio.run(predicted_qca_generator.agenerate_predicted_qca_dataset())
#     evaluator = _create_base_evaluator(config, name)
#     qcas = evaluator.load_predicted_qca_dataset()
#     retrieval_result = asyncio.run(evaluator.aevaluation_for_retrieval(qcas))
#     response_result = asyncio.run(evaluator.aevaluation_for_response(qcas))
#     return retrieval_result, response_result
