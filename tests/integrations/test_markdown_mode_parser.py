from pai_rag.integrations.nodeparsers.base import MarkdownNodeParser
import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.integrations.readers.pai_pdf_reader import PaiPDFReader
from llama_index.core import SimpleDirectoryReader

BASE_DIR = Path(__file__).parent.parent.parent


def test_markdown_parser():
    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfiguration.from_file(config_file).get_value()
    module_registry.init_modules(config)
    ModelScopeDownloader().load_basic_models()
    ModelScopeDownloader().load_mineru_config()
    directory_reader = SimpleDirectoryReader(
        input_dir="tests/testdata/data/pdf_data",
        file_extractor={".pdf": PaiPDFReader()},
    )
    documents = directory_reader.load_data()
    md_node_parser = MarkdownNodeParser(enable_multimodal=False)
    splitted_nodes = []
    for doc_node in documents:
        splitted_nodes.extend(md_node_parser.get_nodes_from_documents([doc_node]))

    assert len(splitted_nodes) == 6
