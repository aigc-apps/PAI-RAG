import os
import json
from typing import Dict, Any
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import DOC_TYPES_CONVERT_TO_MD
from pai_rag.integrations.readers.pai.constants import ACCEPTABLE_DOC_TYPES
from pai_rag.utils.index_utils import (
    delete_dir,
    delete_file,
    write_markdown_to_parse_dir,
    copy_original_files_to_parse_dir,
)
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
    def create_new_knowledgebase_dir(knowledgebase_paths: Dict[str, str]):
        try:
            os.makedirs(knowledgebase_paths["base_path"], exist_ok=True)
            os.makedirs(knowledgebase_paths["docs_path"], exist_ok=True)
            os.makedirs(knowledgebase_paths["index_path"], exist_ok=True)
            os.makedirs(knowledgebase_paths["logs_path"], exist_ok=True)
            logger.info(f"知识库目录 {knowledgebase_paths['base_path']} 及其子目录已成功创建或已存在。")
        except Exception as e:
            logger.error(f"创建目录knowledgebase_paths:{knowledgebase_paths}时发生错误: {e} ")

    @staticmethod
    def get_docid_from_index_via_file_name(doc_ids_map_file, file_name):
        with open(doc_ids_map_file, "r") as f:
            doc_ids_map_dict = json.load(f)
        return doc_ids_map_dict.get(file_name, None)

    @staticmethod
    def delete_local_files_from_index(knowledgebase_paths, file_path):
        file_name = str(file_path).split("/")[-1]
        relative_path = "/".join(file_path.split("/")[4:-1])
        file_type = os.path.splitext(file_name)[1]
        parse_file = os.path.join(
            knowledgebase_paths["parse_path"], relative_path, file_name
        )
        split_path = os.path.join(
            knowledgebase_paths["split_path"], relative_path, file_name
        )
        embed_path = os.path.join(
            knowledgebase_paths["embed_path"], relative_path, file_name
        )

        file_type = f".{parse_file.split('.')[-1]}"
        if file_type in DOC_TYPES_CONVERT_TO_MD:
            parse_file = f"{parse_file}.md"
            logger.debug(f"file {parse_file} is converted to md")
        delete_file(parse_file)
        delete_dir(split_path)
        delete_dir(embed_path)

        if os.path.exists(knowledgebase_paths["doc_ids_map_file"]):
            with open(knowledgebase_paths["doc_ids_map_file"], "r") as json_file:
                try:
                    doc_ids_map_dict = json.load(json_file)
                except json.JSONDecodeError:
                    doc_ids_map_dict = {}
        del doc_ids_map_dict[file_path]
        logger.info(
            f"Deleted file_path: {file_path} from doc_ids_map_dict {doc_ids_map_dict}"
        )
        try:
            with open(knowledgebase_paths["doc_ids_map_file"], "w") as f:
                json.dump(doc_ids_map_dict, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"写入文件{knowledgebase_paths['doc_ids_map_file']}时出错: {e}")

    @staticmethod
    def delete_local_dir_from_index(knowledgebase_paths, file_path):
        relative_path = "/".join(file_path.split("/")[4:])
        parse_dir = os.path.join(knowledgebase_paths["parse_path"], relative_path)
        split_path = os.path.join(knowledgebase_paths["split_path"], relative_path)
        embed_path = os.path.join(knowledgebase_paths["embed_path"], relative_path)

        delete_dir(parse_dir)
        delete_dir(split_path)
        delete_dir(embed_path)

    @staticmethod
    def save_parse_files(knowledgebase_paths, documents):
        doc_ids_map_dict = {}
        if os.path.exists(knowledgebase_paths["doc_ids_map_file"]):
            with open(knowledgebase_paths["doc_ids_map_file"], "r") as json_file:
                try:
                    doc_ids_map_dict = json.load(json_file)
                except json.JSONDecodeError:
                    doc_ids_map_dict = {}

        for doc in documents:
            file_name = doc.metadata.get("file_name", "dummy.none")
            file_path = doc.metadata.get("file_path", None)
            doc_ids_map_dict[file_path] = doc.id_
            file_type = os.path.splitext(file_name)[1]
            relative_path = "/".join(file_path.split("/")[4:-1])
            relative_parse_path = os.path.join(
                knowledgebase_paths["parse_path"], relative_path
            )
            os.makedirs(relative_parse_path, exist_ok=True)
            if file_type in DOC_TYPES_CONVERT_TO_MD:
                write_markdown_to_parse_dir(
                    doc.text, doc.metadata.get("file_name", None), relative_parse_path
                )
            elif file_type in ACCEPTABLE_DOC_TYPES:
                copy_original_files_to_parse_dir(file_path, relative_parse_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
            logger.debug("doc_ids_map_dict", doc_ids_map_dict)

            try:
                with open(knowledgebase_paths["doc_ids_map_file"], "w") as f:
                    json.dump(doc_ids_map_dict, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error(f"写入文件{knowledgebase_paths['doc_ids_map_file']}时出错: {e}")

    @staticmethod
    def save_chunk_nodes(knowledgebase_paths, nodes, operation):
        chunk_path = os.path.join(knowledgebase_paths["index_path"], operation)
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
