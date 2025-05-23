def test_pai_pdf_reader():
    from pai_rag.file.readers.pai.pai_data_reader import (
        PaiDataReader,
        BaseDataReaderConfig,
    )
    from pai_rag.file.readers.pai.file_readers.pai_pdf_reader import PaiPDFReader
    from pai_rag.utils.download_models import ModelScopeDownloader

    reader_config = BaseDataReaderConfig()
    directory_reader = PaiDataReader(reader_config=reader_config)

    input_dir = "tests/testdata/data/pdf_data"
    ModelScopeDownloader().load_rag_models()

    directory_reader.file_readers[".pdf"] = PaiPDFReader()

    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    assert len(documents) > 0


def test_is_horizontal_table():
    from pai_rag.file.readers.pai.utils.markdown_utils import is_horizontal_table

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
    assert is_horizontal_table(horizontal_table_1)
    assert is_horizontal_table(horizontal_table_2)
    assert is_horizontal_table(horizontal_table_3)
    assert not is_horizontal_table(vertical_table)
