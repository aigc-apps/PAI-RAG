import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry
from pai_rag.integrations.readers.pai_pdf_reader import PaiPDFReader
from llama_index.core import SimpleDirectoryReader
import pytest

BASE_DIR = Path(__file__).parent.parent.parent
def test_pai_pdf_reader():
    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfiguration.from_file(config_file).get_value()
    module_registry.init_modules(config)
    reader_config = config["data_reader"]
    directory_reader = SimpleDirectoryReader(
        input_dir="tests/testdata/data/pdf_data",
        file_extractor={
            ".pdf": PaiPDFReader(
                enable_image_ocr=reader_config.get("enable_image_ocr", False),
                model_dir=reader_config.get("easyocr_model_dir", None),
            )
        },
    )
    documents = directory_reader.load_data()
    assert len(documents) > 0


