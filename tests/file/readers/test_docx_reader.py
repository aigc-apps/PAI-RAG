from pai_rag.file.readers.pai.pai_data_reader import PaiDataReader, DataReaderConfig
from pai_rag.file.readers.pai.file_readers.pai_docx_reader import PaiDocxReader


def test_pai_docx_reader():
    reader_config = DataReaderConfig()
    directory_reader = PaiDataReader(reader_config=reader_config)
    input_dir = "tests/testdata/data/docx_data"

    directory_reader.file_readers[".docx"] = PaiDocxReader()

    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    assert "步骤一：部署RAG服务" in str(documents[0].text)
