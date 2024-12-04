from datetime import datetime
from llama_index.core.schema import TextNode, ImageNode


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


def convert_nodes_to_dict(input_nodes):
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

    for node in input_nodes:
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
        data["sparse_embedding"].append(None),
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
