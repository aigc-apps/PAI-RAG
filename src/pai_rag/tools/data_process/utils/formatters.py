from llama_index.core.schema import Document
from datetime import datetime
from llama_index.core.schema import TextNode, ImageNode
import numpy as np
import json


def convert_node_to_dict(node):
    return {
        "id": getattr(node, "id_", None),
        "type": "text" if type(node) is TextNode else "image",
        "text": getattr(node, "text", "") if type(node) is TextNode else "",
        "metadata": {
            k: str(v) if isinstance(v, datetime) else v
            for k, v in getattr(node, "metadata", {}).items()
        },
        "embedding": getattr(node, "embedding", None),
        "sparse_embedding": None,
        "mimetype": getattr(node, "mimetype", None),
        "start_char_idx": getattr(node, "start_char_idx", None),
        "end_char_idx": getattr(node, "end_char_idx", None),
        "text_template": getattr(node, "text_template", None)
        if type(node) is TextNode
        else None,
        "metadata_template": getattr(node, "metadata_template", None),
        "metadata_separator": getattr(node, "metadata_separator", None),
        "excluded_embed_metadata_keys": getattr(
            node, "excluded_embed_metadata_keys", []
        ),
        "excluded_llm_metadata_keys": getattr(node, "excluded_llm_metadata_keys", []),
        # Image-specific attributes
        "image": getattr(node, "image", None) if type(node) is ImageNode else None,
        "image_path": getattr(node, "image_path", None)
        if type(node) is ImageNode
        else None,
        "image_url": getattr(node, "image_url", None)
        if type(node) is ImageNode
        else None,
        "image_mimetype": getattr(node, "image_mimetype", None)
        if type(node) is ImageNode
        else None,
    }


def convert_document_to_dict(doc):
    return {
        "id": doc.id_,
        "embedding": doc.embedding,
        "metadata": doc.metadata,
        "excluded_embed_metadata_keys": doc.excluded_embed_metadata_keys,
        "excluded_llm_metadata_keys": doc.excluded_llm_metadata_keys,
        "relationships": doc.relationships,
        "text": doc.text,
        "mimetype": doc.mimetype,
    }


def convert_list_to_documents(doc_list):
    documents = []
    for doc in doc_list:
        document = Document(
            id_=doc["id"],
            embedding=doc["embedding"],
            metadata=doc["metadata"],
            excluded_embed_metadata_keys=list(doc["excluded_embed_metadata_keys"]),
            excluded_llm_metadata_keys=list(doc["excluded_llm_metadata_keys"]),
            relationships=doc["relationships"],
            text=doc["text"],
            mimetype=doc["mimetype"],
        )
        documents.append(document)
    return documents


def convert_nodes_to_list(input_nodes):
    converted_nodes = [convert_node_to_dict(node) for node in input_nodes]
    return converted_nodes


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
