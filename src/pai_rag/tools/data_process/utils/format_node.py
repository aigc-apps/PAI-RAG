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


def convert_dict_to_nodes(node_dict: dict):
    length = len(node_dict["id"])
    text_nodes = []
    image_nodes = []
    for i in range(length):
        metadata = {
            k: datetime.fromisoformat(v)
            if k == "creation_date" and isinstance(v, str)
            else v
            for k, v in node_dict["metadata"][i].items()
        }
        if node_dict["type"][i] == "text":
            node = TextNode(
                id_=node_dict["id"][i],
                text=node_dict["text"][i],
                metadata=metadata,
                embedding=node_dict["embedding"][i],
                mimetype=node_dict["mimetype"][i],
                start_char_idx=node_dict["start_char_idx"][i],
                end_char_idx=node_dict["end_char_idx"][i],
                text_template=node_dict["text_template"][i],
                metadata_template=node_dict["metadata_template"][i],
                metadata_seperator=node_dict["metadata_seperator"][i],
                excluded_embed_metadata_keys=list(
                    node_dict["excluded_embed_metadata_keys"][i]
                ),
                excluded_llm_metadata_keys=list(
                    node_dict["excluded_llm_metadata_keys"][i]
                ),
            )
            text_nodes.append(node)
        elif node_dict["type"][i] == "image":
            node = ImageNode(
                id_=node_dict["id"][i],
                image=node_dict["image"][i],
                image_path=node_dict["image_path"][i],
                image_url=node_dict["image_url"][i],
                image_mimetype=node_dict["image_mimetype"][i],
                metadata=metadata,
                embedding=node_dict["embedding"][i],
                mimetype=node_dict["mimetype"][i],
                start_char_idx=node_dict["start_char_idx"][i],
                end_char_idx=node_dict["end_char_idx"][i],
                metadata_template=node_dict["metadata_template"][i],
                metadata_seperator=node_dict["metadata_seperator"][i],
                excluded_embed_metadata_keys=list(
                    node_dict["excluded_embed_metadata_keys"][i]
                ),
                excluded_llm_metadata_keys=list(
                    node_dict["excluded_llm_metadata_keys"][i]
                ),
            )
            image_nodes.append(node)
        else:
            raise ValueError(f"Invalid node type: {node_dict['type'][i]}")
    return text_nodes, image_nodes


def convert_nodes_to_dict(text_embed_nodes, text_sparse_embeddings, image_embed_nodes):
    data = {
        "id": [],
        "type": [],
        "text": [],
        "metadata": [],
        "embedding": [],
        "sparse_embedding": [],
        "mimetype": [],
        "start_char_idx": [],
        "end_char_idx": [],
        "text_template": [],
        "metadata_template": [],
        "metadata_seperator": [],
        "excluded_embed_metadata_keys": [],
        "excluded_llm_metadata_keys": [],
        # image
        "image": [],
        "image_path": [],
        "image_url": [],
        "image_mimetype": [],
    }

    embed_nodes = text_embed_nodes + image_embed_nodes
    for idx, node in enumerate(embed_nodes):
        data["id"].append(node.id_)
        data["type"].append("text" if type(node) is TextNode else "image")
        data["text"].append(node.text if type(node) is TextNode else "")
        data["metadata"].append(
            {
                k: str(v) if isinstance(v, datetime) else v
                for k, v in node.metadata.items()
            }
        ),
        data["embedding"].append(node.embedding),
        data["sparse_embedding"].append(
            text_sparse_embeddings[idx]
            if (type(node) is TextNode and text_sparse_embeddings)
            else None
        ),
        data["mimetype"].append(node.mimetype),
        data["start_char_idx"].append(node.start_char_idx),
        data["end_char_idx"].append(node.end_char_idx),
        data["text_template"].append(
            node.text_template if type(node) is TextNode else None
        ),
        data["metadata_template"].append(node.metadata_template),
        data["metadata_seperator"].append(node.metadata_seperator),
        data["excluded_embed_metadata_keys"].append(node.excluded_embed_metadata_keys),
        data["excluded_llm_metadata_keys"].append(node.excluded_llm_metadata_keys),
        # image
        data["image"].append(node.image if type(node) is ImageNode else None),
        data["image_path"].append(node.image_path if type(node) is ImageNode else None),
        data["image_url"].append(node.image_url if type(node) is ImageNode else None),
        data["image_mimetype"].append(
            node.image_mimetype if type(node) is ImageNode else None
        )
    return data
