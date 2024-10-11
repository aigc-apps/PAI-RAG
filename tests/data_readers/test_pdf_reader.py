import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry
from pai_rag.integrations.readers.pai_pdf_reader import PaiPDFReader
from pai_rag.utils.download_models import ModelScopeDownloader
from llama_index.core import SimpleDirectoryReader

BASE_DIR = Path(__file__).parent.parent.parent


def test_pai_pdf_reader():
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
    assert len(documents) > 0


def test_is_horizontal_table():
    # example data
    horizontal_table_1 = [
        ["Name", "Age", "City"],
        ["Alice", 30, "New York"],
        ["Bob", 25, "San Francisco"],
    ]

    horizontal_table_2 = [
        ["Name", "Age", "discount"],
        ["Alice", 30, 0.3],
        ["Bob", 25, 0.4],
    ]

    horizontal_table_3 = [
        ["Age", "discount", "amount"],
        [30, 0.3, 3],
        [25, 0.4, 7],
        [34, 0.2, 9],
    ]

    vertical_table = [
        ["Field", "Record1", "Record2"],
        ["Name", "Alice", "Bob"],
        ["Age", 30, 25],
        ["City", "New York", "San Francisco"],
    ]
    assert PaiPDFReader.is_horizontal_table(horizontal_table_1)
    assert PaiPDFReader.is_horizontal_table(horizontal_table_2)
    assert PaiPDFReader.is_horizontal_table(horizontal_table_3)
    assert not PaiPDFReader.is_horizontal_table(vertical_table)
