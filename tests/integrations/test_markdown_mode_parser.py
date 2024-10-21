from pai_rag.core.rag_module import resolve
from pai_rag.integrations.nodeparsers.base import MarkdownNodeParser
import os
from pathlib import Path
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.utils.download_models import ModelScopeDownloader

BASE_DIR = Path(__file__).parent.parent.parent


def test_markdown_parser():
    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfigManager.from_file(config_file).get_value()
    directory_reader = resolve(
        cls=PaiDataReader,
        reader_config=config.data_reader,
    )
    input_dir = "tests/testdata/data/pdf_data"
    ModelScopeDownloader().load_basic_models()
    ModelScopeDownloader().load_mineru_config()
    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    md_node_parser = MarkdownNodeParser(enable_multimodal=False)
    splitted_nodes = []
    for doc_node in documents:
        splitted_nodes.extend(md_node_parser.get_nodes_from_documents([doc_node]))

    assert len(splitted_nodes) == 6
