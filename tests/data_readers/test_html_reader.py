import os
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).parent.parent.parent


@pytest.mark.skipif(
    os.getenv("SKIP_GPU_TESTS", "false") == "true",
    reason="Need to execute in a CUDA environment.",
)
def test_pai_html_reader():
    from pai_rag.core.rag_config_manager import RagConfigManager
    from pai_rag.core.rag_module import resolve
    from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
    from pai_rag.integrations.readers.pai_html_reader import PaiHtmlReader

    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfigManager.from_file(config_file).get_value()
    directory_reader = resolve(
        cls=PaiDataReader,
        reader_config=config.data_reader,
    )
    input_dir = "tests/testdata/data/html_data"

    directory_reader.file_readers[".html"] = PaiHtmlReader()

    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    assert len(documents) == 5
