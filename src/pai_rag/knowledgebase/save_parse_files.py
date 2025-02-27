import os
import shutil
from pai_rag.knowledgebase.constants import DEFAULT_KNOWLEDGE_PATH
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import DOC_TYPES_CONVERT_TO_MD
from pai_rag.integrations.readers.pai.constants import ACCEPTABLE_DOC_TYPES


def write_markdown_to_parse_dir(md_content, file_name, parse_dir):
    destination_md_file = f"{parse_dir}/{file_name}.md"
    try:
        with open(destination_md_file, "w", encoding="utf-8") as md_file:
            md_file.write(md_content)
        print(f"成功写入{destination_md_file}")
    except IOError as e:
        print(f"写入文件时出错: {e}")


def copy_original_files_to_parse_dir(file_path, parse_dir):
    try:
        if os.path.isfile(file_path):
            shutil.copy(file_path, parse_dir)
            print(f"已复制: {file_path} 到 {parse_dir}")
        else:
            print(f"源文件不存在: {file_path}")
    except Exception as e:
        print(f"复制文件时出错: {file_path} -> {e}")


def save_parse_files(index_name, documents):
    parse_folder = os.path.join(DEFAULT_KNOWLEDGE_PATH, index_name, ".index", "parse")
    os.makedirs(parse_folder, exist_ok=True)
    for doc in documents:
        file_name = doc.metadata.get("file_name", "dummy.none")
        file_type = os.path.splitext(file_name)[1]
        if file_type in DOC_TYPES_CONVERT_TO_MD:
            write_markdown_to_parse_dir(
                doc.text, doc.metadata.get("file_name", None), parse_folder
            )
        elif file_type in ACCEPTABLE_DOC_TYPES:
            copy_original_files_to_parse_dir(
                doc.metadata.get("file_path", None), parse_folder
            )
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
