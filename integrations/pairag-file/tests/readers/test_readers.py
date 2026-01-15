import json
from pairag.file.store.file_store_helper import create_file_store_from_env
from pairag.file.nodeparsers.file_parser import FileParser
from pairag.file.models.file_item import FileItem
from pairag.file.utils.image_caption_tool import ImageCaptionTool
from pairag.file.utils.multimodal_llm import OpenAIMultimodalLLM
import os
import pytest

def test_csv_reader():
    file_store = create_file_store_from_env()

    input_file = "tests/testdata/csv_data/titanic_train.csv"
    file_parser = FileParser(file_store=file_store)
    file_item = FileItem.from_path(
                file_path=input_file,
                kb_id="kb_id",
            )
    documents = file_parser.read_file(file_item,is_attachment=False)
    assert len(documents) == 891

def test_docx_reader():
    file_store = create_file_store_from_env()

    input_file = "tests/testdata/docx_data/大模型RAG对话系统.docx"
    file_parser = FileParser(file_store=file_store)
    file_item = FileItem.from_path(
                file_path=input_file,
                kb_id="kb_id",
            )
    documents = file_parser.read_file(file_item,is_attachment=False)
    assert "步骤一：部署RAG服务" in str(documents[0].text)

def test_excel_reader():
    file_store = create_file_store_from_env()

    input_file = "tests/testdata/excel_data/30天留存率_excel_two_header.xlsx"
    file_parser = FileParser(file_store=file_store)
    file_item = FileItem.from_path(
                file_path=input_file,
                kb_id="kb_id",
            )
    documents = file_parser.read_file(file_item,is_attachment=False)
    assert len(documents) == 8

def test_html_reader():
    file_store = create_file_store_from_env()

    input_file = "tests/testdata/html_data/多媒体分析计费说明.html"
    file_parser = FileParser(file_store=file_store)
    file_item = FileItem.from_path(
                file_path=input_file,
                kb_id="kb_id",
            )
    documents = file_parser.read_file(file_item,is_attachment=False)
    assert len(documents) == 1


# 定义跳过条件
SKIP_REASON = "DASHSCOPE_API_KEY not set"
SKIP_CONDITION = os.environ.get("DASHSCOPE_API_KEY") is None

@pytest.fixture
@pytest.mark.skipif(SKIP_CONDITION, reason=SKIP_REASON)
def image_caption_tool() -> ImageCaptionTool:
    multimodal_llm = OpenAIMultimodalLLM(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        model="qwen-vl-max",
    )
    image_caption_tool = ImageCaptionTool(multimodal_llm=multimodal_llm)
    return image_caption_tool
    
@pytest.mark.skipif(SKIP_CONDITION, reason=SKIP_REASON)
def test_image_reader(image_caption_tool):
    file_store = create_file_store_from_env()

    input_file = "tests/testdata/image_data/11.jpg"
    file_parser = FileParser(file_store=file_store, image_caption_tool=image_caption_tool)
    file_item = FileItem.from_path(
                file_path=input_file,
                kb_id="kb_id",
            )
    documents = file_parser.read_file(file_item,is_attachment=False)
    assert len(documents) == 1

def test_jsonl_reader():
    file_store = create_file_store_from_env()

    input_file = "tests/testdata/jsonl_data/price.jsonl"
    file_parser = FileParser(file_store=file_store)
    file_item = FileItem.from_path(
                file_path=input_file,
                kb_id="kb_id",
            )
    documents = file_parser.read_file(file_item,is_attachment=False)
    assert len(documents) == 27

def test_pdf_reader():
    file_store = create_file_store_from_env()

    input_file = "tests/testdata/pdf_data/pai_document.pdf"
    file_parser = FileParser(file_store=file_store)
    file_item = FileItem.from_path(
                file_path=input_file,
                kb_id="kb_id",
            )
    documents = file_parser.read_file(file_item,is_attachment=False)
    assert len(documents) == 1

def test_pptx_reader():
    file_store = create_file_store_from_env()

    input_file = "tests/testdata/pptx_data/云栖-企业级RAG方案.pptx"
    file_parser = FileParser(file_store=file_store)
    file_item = FileItem.from_path(
                file_path=input_file,
                kb_id="kb_id",
            )
    documents = file_parser.read_file(file_item,is_attachment=False)
    assert len(documents) == 1

def test_text_reader():
    file_store = create_file_store_from_env()

    input_file = "tests/testdata/text_data/rag测试.txt"
    file_parser = FileParser(file_store=file_store)
    file_item = FileItem.from_path(
                file_path=input_file,
                kb_id="kb_id",
            )
    documents = file_parser.read_file(file_item,is_attachment=False)
    assert len(documents) == 1