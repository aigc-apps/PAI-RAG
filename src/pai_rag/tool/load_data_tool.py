import click
import os
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
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_BASE_DIR = Path(__file__).parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")


def _create_data_loader(config_file, enable_raptor: bool = False) -> RagDataLoader:
    config = RagConfiguration.from_file(config_file).get_value()

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

    return data_loader


@click.command()
@click.option(
    "-c",
    "--config",
    show_default=True,
    help=f"Configuration file. Default: {DEFAULT_APPLICATION_CONFIG_FILE}",
    default=DEFAULT_APPLICATION_CONFIG_FILE,
)
@click.option(
    "-o",
    "--oss_path",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="oss path (file or directory) to ingest. Example: oss://rag-demo/testdata",
)
@click.option(
    "-d",
    "--data_path",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="data path (file or directory) to ingest.",
)
@click.option(
    "-p",
    "--pattern",
    required=False,
    type=str,
    default=None,
    help="data pattern to ingest.",
)
@click.option(
    "-r",
    "--enable_raptor",
    required=False,
    is_flag=True,
    show_default=True,
    default=False,
    help="use raptor for node enhancement.",
)
def run(
    config=None,
    oss_path=None,
    data_path=None,
    pattern=None,
    enable_raptor=False,
):
    assert (oss_path is not None) or (
        data_path is not None
    ), "Must provide either local path or oss path."
    assert (oss_path is None) or (
        data_path is None
    ), f"Can not provide both local path '{data_path}' and oss path '{oss_path}'."

    data_loader = _create_data_loader(config, enable_raptor)
    data_loader.load_data(
        file_path_or_directory=data_path,
        filter_pattern=pattern,
        oss_path=oss_path,
        from_oss=oss_path is not None,
        enable_raptor=enable_raptor,
    )
