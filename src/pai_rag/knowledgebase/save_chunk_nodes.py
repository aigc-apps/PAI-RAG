import os
import json
from typing import Dict, Any
from pai_rag.knowledgebase.constants import DEFAULT_KNOWLEDGE_PATH

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


def save_chunk_nodes(index_name, nodes, operation):
    chunk_folder = os.path.join(DEFAULT_KNOWLEDGE_PATH, index_name, ".index", operation)
    os.makedirs(chunk_folder, exist_ok=True)
    file_name_dict = {}
    for node in nodes:
        file_name = node.metadata.get("file_name", "dummy.none")
        if file_name in file_name_dict:
            file_name_dict[file_name] += 1
        else:
            file_name_dict[file_name] = 1
        file_chunk_dir = os.path.join(chunk_folder, file_name)
        os.makedirs(file_chunk_dir, exist_ok=True)
        file_path = f"{file_chunk_dir}/{file_name_dict[file_name]}.json"
        with open(file_path, mode="w", encoding="utf-8") as file:
            json.dump(filter_dict(node.dict()), file, ensure_ascii=False, indent=4)
