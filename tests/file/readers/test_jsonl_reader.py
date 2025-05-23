def test_jsonl_reader():
    from pai_rag.file.readers.pai.pai_data_reader import (
        PaiDataReader,
        DataReaderConfig,
    )
    from pai_rag.file.readers.pai.file_readers.pai_jsonl_reader import PaiJsonLReader

    reader_config = DataReaderConfig()
    directory_reader = PaiDataReader(reader_config=reader_config)

    input_dir = "tests/testdata/data/jsonl_data"
    directory_reader.file_readers[".jsonl"] = PaiJsonLReader()

    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    for doc in documents:
        print(doc)
    assert len(documents) == 27
