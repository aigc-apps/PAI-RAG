import os
import json
from typing import Dict, Any
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import DOC_TYPES_CONVERT_TO_MD
from pai_rag.integrations.readers.pai.constants import ACCEPTABLE_DOC_TYPES
from pai_rag.utils.index_utils import (
    write_markdown_to_parse_dir,
    copy_original_files_to_parse_dir,
)
from pai_rag.utils.constants import DEFAULT_KNOWLEDGEBASE_PATH
from loguru import logger

EXCLUDE_NODE_KEYS = set(
    [
        "relationships",
        "excluded_embed_metadata_keys",
        "excluded_llm_metadata_keys",
        "metadata_template",
        "metadata_separator",
        "mimetype",
        "start_char_idx",
        "end_char_idx",
        "metadata_seperator",
        "text_template",
    ]
)


def filter_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """排除指定键的函数"""
    return {k: v for k, v in data.items() if k not in EXCLUDE_NODE_KEYS}


class RagKnowledgeBaseHelper:
    @staticmethod
    def create_new_knowledgebase_dir(knowledgebase_name: str):
        try:
            base_path = os.path.join(DEFAULT_KNOWLEDGEBASE_PATH, knowledgebase_name)
            docs_path = os.path.join(
                DEFAULT_KNOWLEDGEBASE_PATH, knowledgebase_name, "docs"
            )
            index_path = os.path.join(
                DEFAULT_KNOWLEDGEBASE_PATH, knowledgebase_name, ".index"
            )
            os.makedirs(base_path, exist_ok=True)
            os.makedirs(docs_path, exist_ok=True)
            os.makedirs(index_path, exist_ok=True)
            logger.info(f"知识库 {knowledgebase_name} 及其子目录已成功创建或已存在。")
        except Exception as e:
            logger.error(f"创建知识库 {knowledgebase_name}时发生错误: {e} ")

    @staticmethod
    def save_parse_files(knowledgebase_name, documents):
        parse_path = os.path.join(
            DEFAULT_KNOWLEDGEBASE_PATH, knowledgebase_name, ".index", "parse"
        )
        ORIGINAL_DOCS_MAP = dict()
        for doc in documents:
            file_name = doc.metadata.get("file_name", "dummy.none")
            file_path = doc.metadata.get("file_path", None)
            file_type = os.path.splitext(file_name)[1]
            relative_path = "/".join(file_path.split("/")[4:-1])

            relative_parse_path = os.path.join(parse_path, relative_path)
            os.makedirs(relative_parse_path, exist_ok=True)
            if file_type in DOC_TYPES_CONVERT_TO_MD:
                write_markdown_to_parse_dir(
                    doc.text, doc.metadata.get("file_name", None), relative_parse_path
                )
            elif file_type in ACCEPTABLE_DOC_TYPES:
                if file_path not in ORIGINAL_DOCS_MAP:
                    ORIGINAL_DOCS_MAP[file_path] = relative_parse_path
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
        # copy original excel/jsonl files to parse dir only once
        if len(ORIGINAL_DOCS_MAP) > 0:
            for file_path, relative_parse_path in ORIGINAL_DOCS_MAP.items():
                copy_original_files_to_parse_dir(file_path, relative_parse_path)

    @staticmethod
    def save_chunk_nodes(knowledgebase_name, nodes, operation):
        index_path = os.path.join(
            DEFAULT_KNOWLEDGEBASE_PATH, knowledgebase_name, ".index"
        )
        chunk_path = os.path.join(index_path, operation)
        os.makedirs(chunk_path, exist_ok=True)
        file_name_dict = {}
        for node in nodes:
            file_name = node.metadata.get("file_name", "dummy.none")
            if file_name in file_name_dict:
                file_name_dict[file_name] += 1
            else:
                file_name_dict[file_name] = 1
            file_path = node.metadata.get("file_path", None)
            relative_path = "/".join(file_path.split("/")[4:-1])
            file_chunk_dir = os.path.join(chunk_path, relative_path, file_name)
            os.makedirs(file_chunk_dir, exist_ok=True)
            node_file_path = os.path.join(
                file_chunk_dir, f"{file_name_dict[file_name]}.json"
            )
            with open(node_file_path, mode="w", encoding="utf-8") as file:
                json.dump(filter_dict(node.dict()), file, ensure_ascii=False, indent=4)
