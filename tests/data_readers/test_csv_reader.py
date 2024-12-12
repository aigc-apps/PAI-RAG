import os
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).parent.parent.parent


pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_GPU_TESTS", "false") == "true",
    reason="Need to execute in a CUDA environment.",
)


@pytest.fixture(scope="module", autouse=True)
def test_csv_reader():
    from pai_rag.core.rag_config_manager import RagConfigManager
    from pai_rag.core.rag_module import resolve
    from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
    from pai_rag.integrations.readers.pai_csv_reader import PaiCSVReader

    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfigManager.from_file(config_file).get_value()
    directory_reader = resolve(
        cls=PaiDataReader,
        reader_config=config.data_reader,
    )
    input_dir = "tests/testdata/data/csv_data"
    directory_reader.file_readers[".csv"] = PaiCSVReader(
        concat_rows=config.data_reader.concat_csv_rows,
        header=[0, 1],
    )

    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    for doc in documents:
        print(doc)
    assert len(documents) == 897


def test_pandas_csv_reader():
    from pai_rag.core.rag_config_manager import RagConfigManager
    from pai_rag.core.rag_module import resolve
    from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
    from pai_rag.integrations.readers.pai_csv_reader import PaiPandasCSVReader

    config_file = os.path.join(BASE_DIR, "src/pai_rag/config/settings.toml")
    config = RagConfigManager.from_file(config_file).get_value()
    directory_reader = resolve(
        cls=PaiDataReader,
        reader_config=config.data_reader,
    )
    input_dir = "tests/testdata/data/csv_data"
    directory_reader.file_readers[".csv"] = PaiPandasCSVReader(
        concat_rows=config.data_reader.concat_csv_rows,
        pandas_config={"header": [0, 1]},
    )
    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    for doc in documents:
        print(doc)
    assert len(documents) == 897
