from datetime import datetime
from llama_index.core.schema import TextNode


def text_node_to_dict(node):
    return {
        "id": node.id_,
        "embedding": node.embedding,
        "metadata": {
            k: str(v) if isinstance(v, datetime) else v
            for k, v in node.metadata.items()
        },
        "text": node.text,
        "mimetype": node.mimetype,
        "start_char_idx": node.start_char_idx,
        "end_char_idx": node.end_char_idx,
        "text_template": node.text_template,
        "metadata_template": node.metadata_template,
        "metadata_seperator": node.metadata_seperator,
    }


# 从字典创建 TextNode 对象的函数
def dict_to_text_node(node_dict):
    # 处理 metadata，确保 datetime 或其他类型的字段恢复原型
    metadata = {
        k: datetime.fromisoformat(v)
        if k == "creation_date" and isinstance(v, str)
        else v
        for k, v in node_dict["metadata"].items()
    }

    return TextNode(
        id_=node_dict["id"],
        embedding=node_dict["embedding"],
        metadata=metadata,
        text=node_dict["text"],
        mimetype=node_dict["mimetype"],
        start_char_idx=node_dict.get("start_char_idx", None),
        end_char_idx=node_dict.get("end_char_idx", None),
        text_template=node_dict.get("text_template", ""),
        metadata_template=node_dict.get("metadata_template", ""),
        metadata_seperator=node_dict.get("metadata_seperator", ""),
    )
