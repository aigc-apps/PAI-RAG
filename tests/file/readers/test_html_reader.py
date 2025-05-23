from pai_rag.file.readers.pai.pai_data_reader import PaiDataReader, BaseDataReaderConfig
from pai_rag.file.readers.pai.file_readers.pai_html_reader import PaiHtmlReader


def test_pai_html_reader():
    directory_reader = PaiDataReader(reader_config=BaseDataReaderConfig())
    input_dir = "tests/testdata/data/html_data"

    directory_reader.file_readers[".html"] = PaiHtmlReader()

    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    assert len(documents) == 5
