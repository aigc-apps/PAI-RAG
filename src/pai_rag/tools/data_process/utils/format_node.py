from datetime import datetime
from llama_index.core.schema import TextNode, ImageNode


def node_to_dict(node, sparse_embedding=None):
    if type(node) is TextNode:
        return {
            "id": node.id_,
            "type": "text",
            "text": node.text,
            "metadata": {
                k: str(v) if isinstance(v, datetime) else v
                for k, v in node.metadata.items()
            },
            "embedding": node.embedding,
            "sparse_embedding": sparse_embedding,
            "mimetype": node.mimetype,
            "start_char_idx": node.start_char_idx,
            "end_char_idx": node.end_char_idx,
            "text_template": node.text_template,
            "metadata_template": node.metadata_template,
            "metadata_seperator": node.metadata_seperator,
            "excluded_embed_metadata_keys": node.excluded_embed_metadata_keys,
            "excluded_llm_metadata_keys": node.excluded_llm_metadata_keys,
        }
    elif type(node) is ImageNode:
        return {
            "id": node.id_,
            "type": "image",
            "image": node.image,
            "image_path": node.image_path,
            "image_url": node.image_url,
            "image_mimetype": node.image_mimetype,
            "metadata": {
                k: str(v) if isinstance(v, datetime) else v
                for k, v in node.metadata.items()
            },
            "embedding": node.embedding,
            "mimetype": node.mimetype,
            "start_char_idx": node.start_char_idx,
            "end_char_idx": node.end_char_idx,
            "metadata_template": node.metadata_template,
            "metadata_seperator": node.metadata_seperator,
            "excluded_embed_metadata_keys": node.excluded_embed_metadata_keys,
            "excluded_llm_metadata_keys": node.excluded_llm_metadata_keys,
        }
    else:
        raise ValueError(f"Invalid node type: {type(node)}")


def dict_to_node(node_dict):
    # 处理 metadata，确保 datetime 或其他类型的字段恢复原型
    metadata = {
        k: datetime.fromisoformat(v)
        if k == "creation_date" and isinstance(v, str)
        else v
        for k, v in node_dict["metadata"].items()
    }
    if node_dict["type"] == "text":
        return TextNode(
            id_=node_dict["id"],
            text=node_dict["text"],
            metadata=metadata,
            embedding=node_dict["embedding"],
            mimetype=node_dict["mimetype"],
            start_char_idx=node_dict.get("start_char_idx", None),
            end_char_idx=node_dict.get("end_char_idx", None),
            text_template=node_dict.get("text_template", ""),
            metadata_template=node_dict.get("metadata_template", ""),
            metadata_seperator=node_dict.get("metadata_seperator", ""),
            excluded_embed_metadata_keys=node_dict.get(
                "excluded_embed_metadata_keys", None
            ),
            excluded_llm_metadata_keys=node_dict.get(
                "excluded_llm_metadata_keys", None
            ),
        )
    elif node_dict["type"] == "image":
        return ImageNode(
            id_=node_dict["id"],
            image=node_dict.get("image", None),
            image_path=node_dict.get("image_path", None),
            image_url=node_dict.get("image_url", None),
            image_mimetype=node_dict.get("image_mimetype", None),
            metadata=metadata,
            embedding=node_dict.get("embedding", None),
            mimetype=node_dict.get("mimetype", None),
            start_char_idx=node_dict.get("start_char_idx", None),
            end_char_idx=node_dict.get("end_char_idx", None),
            metadata_template=node_dict.get("metadata_template", None),
            metadata_seperator=node_dict.get("metadata_seperator", None),
            excluded_embed_metadata_keys=node_dict.get(
                "excluded_embed_metadata_keys", None
            ),
            excluded_llm_metadata_keys=node_dict.get(
                "excluded_llm_metadata_keys", None
            ),
        )
    else:
        raise ValueError(f"Invalid node type: {node_dict['type']}")
