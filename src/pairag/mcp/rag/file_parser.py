import uuid
from loguru import logger
from typing import Dict, List
from llama_index.core.schema import Document, BaseNode, TextNode
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter

from pairag.db.models.knowledgebase.knowledgebase import ChunkConfig, KbEntity
from pairag.mcp.providers.knowledgebase_provider import KnowledgebaseProvider
from pairag.mcp.rag.file.file_utils import ensure_file_type_is_supported
from pairag.mcp.rag.file.models.file_item import FileItem
from pairag.mcp.rag.file.readers.base import BaseReader
from pairag.mcp.rag.file.readers.csv_reader import CsvReader
from pairag.mcp.rag.file.readers.doc_reader import DocxReader
from pairag.mcp.rag.file.readers.excel_reader import ExcelReader
from pairag.mcp.rag.file.readers.html_reader import HtmlReader
from pairag.mcp.rag.file.readers.image_reader import ImageReader
from pairag.mcp.rag.file.readers.jsonl_reader import JsonReader
from pairag.mcp.rag.file.readers.markdown_reader import MarkdownReader
from pairag.mcp.rag.file.readers.pdf_reader import MineruPdfReader
from pairag.mcp.rag.file.readers.pptx_reader import PptxReader
from pairag.mcp.rag.file.readers.text_reader import TextReader
from pairag.mcp.rag.file.splitters.pai_markdown_parser import MarkdownNodeParser
from pairag.mcp.rag.file.store.base import BaseFileStore
from pairag.mcp.rag.image_caption_tool import ImageCaptionTool


DOC_TYPES_DO_NOT_NEED_CHUNKING = set([".csv", ".xlsx", ".xls", ".jsonl"])
DOC_TYPES_CONVERT_TO_MD = set([".md", ".pdf", ".docx", ".htm", ".html", ".pptx"])
DEFAULT_EXCLUDED_METADATA_KEYS = [
    "file_type",
    "file_size",
    "creation_date",
    "last_modified_date",
    "last_accessed_date",
    "file_path",
    "image_url",
    "total_pages",
    "source",
    "row_number",
    "image_info_list",
    "file_url",
    "ref_doc_id",
]


def node_id_func(i: int, doc: BaseNode) -> str:
    return uuid.uuid4().hex


class FileParser:
    def __init__(
        self,
        file_store: BaseFileStore,
        knowledgebase_provider: KnowledgebaseProvider,
        image_caption_tool: ImageCaptionTool = None,
    ):
        self.file_store = file_store
        self.file_readers: Dict[str, BaseReader] = {
            ".md": MarkdownReader(
                file_store=file_store, image_caption_tool=image_caption_tool
            ),
            ".pdf": MineruPdfReader(
                file_store=file_store, image_caption_tool=image_caption_tool
            ),
            ".html": HtmlReader(
                file_store=file_store, image_caption_tool=image_caption_tool
            ),
            ".htm": HtmlReader(
                file_store=file_store, image_caption_tool=image_caption_tool
            ),
            ".docx": DocxReader(
                file_store=file_store, image_caption_tool=image_caption_tool
            ),
            ".pptx": PptxReader(
                file_store=file_store, image_caption_tool=image_caption_tool
            ),
            ".jsonl": JsonReader(),
            ".csv": CsvReader(),
            ".xlsx": ExcelReader(),
            ".xls": ExcelReader(),
            ".jpg": ImageReader(
                file_store=file_store, image_caption_tool=image_caption_tool
            ),
            ".png": ImageReader(
                file_store=file_store, image_caption_tool=image_caption_tool
            ),
            ".jpeg": ImageReader(
                file_store=file_store, image_caption_tool=image_caption_tool
            ),
            ".txt": TextReader(),
        }
        self.knowledgebase_provider = knowledgebase_provider

    # 读取文件解析为Document列表
    def read_file(self, file_item: FileItem) -> List[Document]:
        ensure_file_type_is_supported(file_item.file_extension)

        reader = self.file_readers.get(file_item.file_extension)
        return reader.read(file_item)

    def split_docs(
        self, docs: List[Document], chunk_config: ChunkConfig
    ) -> List[BaseNode]:
        splitted_nodes: List[BaseNode] = []

        for doc_node in docs:
            logger.info(f"Start splitting document: {doc_node.metadata['file_name']}")

            chunks = []
            doc_type = doc_node.metadata["file_extension"]
            if doc_type in DOC_TYPES_DO_NOT_NEED_CHUNKING:
                # 表格格式文档
                metadata = doc_node.metadata

                node_id = uuid.uuid4().hex
                chunks.append(
                    TextNode(
                        id_=node_id,
                        text=doc_node.text,
                        metadata=metadata,
                        relationships={
                            NodeRelationship.SOURCE: RelatedNodeInfo(
                                node_id=doc_node.node_id, metadata={}
                            ),
                        },
                    )
                )
            else:
                parser = SentenceSplitter(
                    id_func=node_id_func,
                    chunk_size=chunk_config.chunk_size,
                    chunk_overlap=chunk_config.chunk_overlap,
                    paragraph_separator=chunk_config.separator,
                )
                if doc_type in DOC_TYPES_CONVERT_TO_MD:
                    # markdown格式(pdf, md, html, doc 等)
                    md_node_parser = MarkdownNodeParser(
                        chunk_size=chunk_config.chunk_size,
                        chunk_overlap_size=chunk_config.chunk_overlap,
                        base_parser=parser,
                    )
                    chunks = md_node_parser.get_nodes_from_documents([doc_node])
                else:
                    # txt格式等纯文本
                    chunks = parser.get_nodes_from_documents([doc_node])

            splitted_nodes.extend(chunks)
            logger.info(
                f"Finished split document into {len(chunks)} chunks: {doc_node.metadata['file_name']}"
            )

        for node in splitted_nodes:
            node.excluded_embed_metadata_keys = list(
                set(node.excluded_embed_metadata_keys + DEFAULT_EXCLUDED_METADATA_KEYS)
            )
            node.excluded_llm_metadata_keys = list(
                set(node.excluded_llm_metadata_keys + DEFAULT_EXCLUDED_METADATA_KEYS)
            )

        logger.info(
            f"[DataReader] Split {len(docs)} documents into {len(splitted_nodes)} nodes."
        )

        return splitted_nodes

    def parse(self, file_item: FileItem) -> List[BaseNode]:
        knowledgebase = self.knowledgebase_provider.get_knowledgebase(file_item.kb_id)
        docs = self.read_file(file_item)
        chunk_config = ChunkConfig.model_validate(knowledgebase.chunk_config)
        nodes = self.split_docs(docs, chunk_config=chunk_config)
        return nodes


if __name__ == "__main__":
    from pairag.integrations.llms.pai.open_ai_alike_multi_modal import (
        OpenAIAlikeMultiModal,
    )
    from pairag.mcp.rag.file.store.oss_store import OssFileStore
    import os

    knowledgebase_provider = KnowledgebaseProvider()
    knowledgebase_provider.knowledgebase_map = {
        "test": KbEntity(
            id="test",
            name="test",
            chunk_config=ChunkConfig(
                chunk_size=1000,
                chunk_overlap=50,
            ),
        )
    }
    pdf_file = "/Users/feiyue/Documents/test_files/舒福德产品说明书.pdf"
    pdf_file_item = FileItem.from_path(pdf_file, knowledgebase_id="test")
    multimodal_llm = OpenAIAlikeMultiModal(
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        model="qwen-vl-max",
        is_chat_model=True,
    )
    image_caption_tool = ImageCaptionTool(multimodal_llm=multimodal_llm)
    oss_store = OssFileStore(bucket="pai-rag", endpoint="oss-cn-hangzhou.aliyuncs.com")
    file_parser = FileParser(
        file_store=oss_store,
        image_caption_tool=image_caption_tool,
        knowledgebase_provider=knowledgebase_provider,
    )
    chunks = file_parser.parse(pdf_file_item)
    for i, chunk in enumerate(chunks):
        print("==== CHUNK ", i)
        print(chunk.text)
        print(chunk.metadata)
        print("-----")
    print("finished.")
