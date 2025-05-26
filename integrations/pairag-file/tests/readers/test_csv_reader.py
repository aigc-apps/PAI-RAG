from pairag.file.readers.pai.pai_data_reader import DataReaderConfig, PaiDataReader
from pairag.file.readers.pai.file_readers.pai_csv_reader import PaiPandasCSVReader


def test_pandas_csv_reader():
    directory_reader = PaiDataReader(reader_config=DataReaderConfig())
    input_dir = "tests/testdata/csv_data"
    directory_reader.file_readers[".csv"] = PaiPandasCSVReader(
        concat_rows=False,
        pandas_config={"header": [0, 1]},
    )
    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    for doc in documents:
        print(doc)
    assert len(documents) == 897
