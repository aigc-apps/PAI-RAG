import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry
from pai_rag.integrations.readers.pai_csv_reader import PaiCSVReader, PaiPandasCSVReader
from llama_index.core import SimpleDirectoryReader

BASE_DIR = Path(__file__).parent.parent.parent


def test_csv_reader():
    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfiguration.from_file(config_file).get_value()
    module_registry.init_modules(config)
    reader_config = config["rag"]["data_reader"]
    directory_reader = SimpleDirectoryReader(
        input_dir="tests/testdata/data/csv_data",
        file_extractor={
            ".csv": PaiCSVReader(
                concat_rows=reader_config.get("concat_rows", False),
                header=[0, 1],
            )
        },
    )
    documents = directory_reader.load_data()
    for doc in documents:
        print(doc)
    assert len(documents) == 897


def test_pandas_csv_reader():
    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfiguration.from_file(config_file).get_value()
    module_registry.init_modules(config)
    reader_config = config["rag"]["data_reader"]
    directory_reader = SimpleDirectoryReader(
        input_dir="tests/testdata/data/csv_data",
        file_extractor={
            ".csv": PaiPandasCSVReader(
                concat_rows=reader_config.get("concat_rows", False),
                pandas_config={"header": [0, 1]},
            )
        },
    )
    documents = directory_reader.load_data()
    for doc in documents:
        print(doc)
    assert len(documents) == 897
