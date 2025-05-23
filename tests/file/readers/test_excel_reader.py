def test_pandas_excel_reader():
    from pai_rag.file.readers.pai.pai_data_reader import (
        PaiDataReader,
        BaseDataReaderConfig,
    )
    from pai_rag.file.readers.pai.file_readers.pai_excel_reader import (
        PaiPandasExcelReader,
    )

    reader_config = BaseDataReaderConfig()
    directory_reader = PaiDataReader(reader_config=reader_config)
    input_dir = "tests/testdata/data/excel_data"
    directory_reader.file_readers[".xlsx"] = PaiPandasExcelReader(
        concat_rows=reader_config.concat_csv_rows,
        pandas_config={"header": [0, 1]},
    )
    directory_reader.file_readers[".xls"] = PaiPandasExcelReader(
        concat_rows=reader_config.concat_csv_rows,
        pandas_config={"header": [0, 1]},
    )

    documents = directory_reader.load_data(file_path_or_directory=input_dir)

    for doc in documents:
        print(doc)
    assert len(documents) == 7
