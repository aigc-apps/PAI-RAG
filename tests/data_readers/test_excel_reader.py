import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry
from pai_rag.integrations.readers.pai_excel_reader import PaiPandasExcelReader
from llama_index.core import SimpleDirectoryReader

BASE_DIR = Path(__file__).parent.parent.parent


def test_pandas_excel_reader():
    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfiguration.from_file(config_file).get_value()
    module_registry.init_modules(config)
    reader_config = config["rag"]["data_reader"]
    directory_reader = SimpleDirectoryReader(
        input_dir="tests/testdata/data/excel_data",
        file_extractor={
            ".xlsx": PaiPandasExcelReader(
                concat_rows=reader_config.get("concat_rows", False),
                pandas_config={"header": [0, 1]},
            ),
            ".xls": PaiPandasExcelReader(
                concat_rows=reader_config.get("concat_rows", False),
                pandas_config={"header": [0, 1]},
            ),
        },
    )
    documents = directory_reader.load_data()
    for doc in documents:
        print(doc)
    assert len(documents) == 7
