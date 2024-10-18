import click
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
)
from pai_rag.evaluation.generator.labelled_qca_generator import LabelledRagQcaGenerator
from pai_rag.evaluation.generator.predicted_qca_generator import (
    PredictedRagQcaGenerator,
)
from pai_rag.integrations.llms.pai.pai_multi_modal_llm import (
    PaiMultiModalLlm,
)
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_BASE_DIR = Path(__file__).parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(
    _BASE_DIR, "evaluation/settings_eval.toml"
)


def _create_data_loader(config_file, enable_raptor: bool = False) -> RagDataLoader:
    config = RagConfigManager.from_file(config_file).get_value()
    data_loader = resolve_data_loader(config)
    vector_index = resolve_vector_index(config)

    return data_loader, vector_index


def _create_labelled_qca_generator(config_file, vector_index) -> None:
    config = RagConfigManager.from_file(config_file).get_value()
    llm = resolve_llm(config)
    qca_generator = LabelledRagQcaGenerator(
        llm=llm, vector_index=vector_index, persist_path=config.rag.index.persist_path
    )
    return qca_generator


def _create_predicted_qca_generator(config_file, vector_index) -> None:
    config = RagConfigManager.from_file(config_file).get_value()
    # llm = resolve_llm(config)
    multimodal_llm = resolve(cls=PaiMultiModalLlm, llm_config=config.multimodal_llm)
    predicted_qca_generator = PredictedRagQcaGenerator(
        llm=multimodal_llm,
        vector_index=vector_index,
        persist_path=config.rag.index.persist_path,
    )
    return predicted_qca_generator


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

    data_loader, vector_index = _create_data_loader(config, enable_raptor)
    data_loader.load_data(
        file_path_or_directory=data_path,
        filter_pattern=pattern,
        oss_path=oss_path,
        from_oss=oss_path is not None,
        enable_raptor=enable_raptor,
    )
    qca_generator = _create_labelled_qca_generator(config, vector_index)
    asyncio.run(qca_generator.agenerate_labelled_qca_dataset())

    predicted_qca_generator = _create_predicted_qca_generator(config, vector_index)
    asyncio.run(predicted_qca_generator.agenerate_predicted_qca_dataset())
