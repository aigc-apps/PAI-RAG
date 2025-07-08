import pytest
from pairag.mcp.online_file_readers.pai_online_data_reader import PaiOnlineDataReader


class TestPaiOnlineDataReader:
    @pytest.fixture
    def test_initialization(self):
        data_reader = PaiOnlineDataReader()
        assert len(data_reader.file_readers) == 3

    def test_load_data(self):
        """测试load data方法"""
        data_reader = PaiOnlineDataReader()
        documents = data_reader.load_data(
            file_path_or_directory="tests/testdata/pdf_data"
        )
        assert len(documents) == 1
