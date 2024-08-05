import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry
from pai_rag.integrations.readers.pai_jsonl_reader import PaiJsonLReader
from llama_index.core import SimpleDirectoryReader

BASE_DIR = Path(__file__).parent.parent.parent


def test_jsonl_reader():
    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfiguration.from_file(config_file).get_value()
    module_registry.init_modules(config)
    directory_reader = SimpleDirectoryReader(
        input_dir="tests/testdata/data/jsonl_data",
        file_extractor={".jsonl": PaiJsonLReader()},
    )
    documents = directory_reader.load_data()
    for doc in documents:
        print(doc)
    assert len(documents) == 27
