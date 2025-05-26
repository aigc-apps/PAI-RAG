import os

import pytest
from pairag.file.nodeparsers.pai.image_caption_tool import ImageCaptionTool
from pairag.file.readers.pai.pai_data_reader import PaiDataReader, DataReaderConfig
from pairag.file.nodeparsers.pai.pai_node_parser import PaiNodeParser, NodeParserConfig
from pairag.file.store.oss_store import PaiOssStore
from pairag.file.store.pai_image_store import PaiImageStore

from tests.openailike_multimodal import (
    OpenAIAlikeMultiModal,
)

TEST_FILE_DIRECTORY = "tests/testdata/"


def test_read_documents():
    data_reader = PaiDataReader(DataReaderConfig())
    node_parser = PaiNodeParser(NodeParserConfig())

    docs = data_reader.load_data(
        file_path_or_directory=TEST_FILE_DIRECTORY,
        supported_file_types=[
            ".pdf",
            ".docx",
            ".html",
            ".md",
            ".csv",
            ".xls",
            ".xlsx",
            ".jsonl",
        ],  # 不包含图片文件
        show_progress=True,
    )
    assert len(docs) == 942, "document count should be 943."

    chunks = node_parser.get_nodes_from_documents(docs)
    assert len(chunks) == 1074, "chunk count should be 1074"


def test_read_documents_with_image_but_no_oss_configured():
    data_reader = PaiDataReader(DataReaderConfig())
    try:
        data_reader.load_data(
            file_path_or_directory=TEST_FILE_DIRECTORY,
            show_progress=True,
        )
        assert False, "Should throws here. No OSS configured."
    except Exception as e:
        assert "Oss config must be provided for image processing" in str(
            e
        ), "OSS_ACCESS_KEY_ID not found in environment variables"


if (
    not os.environ.get("OSS_ACCESS_KEY_ID")
    or not os.environ.get("OSS_ACCESS_KEY_SECRET")
    or not os.environ.get("DASHSCOPE_API_KEY")
):
    pytest.skip(
        reason="OSS_ACCESS_KEY_ID or OSS_ACCESS_KEY_SECRET or DASHSCOPE_API_KEY not set",
        allow_module_level=True,
    )


@pytest.fixture
def image_store():
    oss_store = PaiOssStore(
        bucket_name="feiyue-test", endpoint="oss-cn-hangzhou.aliyuncs.com"
    )
    return PaiImageStore(oss_store=oss_store)


@pytest.fixture
def multimodal_llm() -> OpenAIAlikeMultiModal:
    return OpenAIAlikeMultiModal(
        model="qwen-vl-max",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
    )


def test_read_documents_with_image_and_oss_configured(image_store, multimodal_llm):
    data_reader = PaiDataReader(DataReaderConfig(), image_store=image_store)

    image_caption_tool = ImageCaptionTool(multimodal_llm=multimodal_llm)
    node_parser = PaiNodeParser(NodeParserConfig(), caption_tool=image_caption_tool)

    docs = data_reader.load_data(
        file_path_or_directory=TEST_FILE_DIRECTORY,
        show_progress=True,
    )
    assert len(docs) == 943, "document count should be 943."

    chunks = node_parser.get_nodes_from_documents(docs)
    assert len(chunks) == 1074, "chunk count should be 1074"
