from llama_index.core.schema import Document
from datetime import datetime
from llama_index.core.schema import TextNode, ImageNode
import numpy as np


class ConvertedNode:
    def __init__(self, node):
        self.id = getattr(node, "id_", None)
        self.type = "text" if type(node) is TextNode else "image"
        self.text = getattr(node, "text", "") if type(node) is TextNode else ""
        self.metadata = {
            k: str(v) if isinstance(v, datetime) else v
            for k, v in getattr(node, "metadata", {}).items()
        }
        self.embedding = getattr(node, "embedding", None)
        self.sparse_embedding = None
        self.mimetype = getattr(node, "mimetype", None)
        self.start_char_idx = getattr(node, "start_char_idx", None)
        self.end_char_idx = getattr(node, "end_char_idx", None)
        self.text_template = (
            getattr(node, "text_template", None) if type(node) is TextNode else None
        )
        self.metadata_template = getattr(node, "metadata_template", None)
        self.metadata_separator = getattr(node, "metadata_separator", None)
        self.excluded_embed_metadata_keys = np.array(
            getattr(node, "excluded_embed_metadata_keys", [])
        )
        self.excluded_llm_metadata_keys = np.array(
            getattr(node, "excluded_llm_metadata_keys", [])
        )
        # Image-specific attributes
        self.image = getattr(node, "image", None) if type(node) is ImageNode else None
        self.image_path = (
            getattr(node, "image_path", None) if type(node) is ImageNode else None
        )
        self.image_url = (
            getattr(node, "image_url", None) if type(node) is ImageNode else None
        )
        self.image_mimetype = (
            getattr(node, "image_mimetype", None) if type(node) is ImageNode else None
        )


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


def convert_dict_to_documents(doc_dict):
    length = len(doc_dict["id"])
    documents = []
    for i in range(length):
        document = Document(
            id_=doc_dict["id"][i],
            embedding=doc_dict["embedding"][i],
            metadata=doc_dict["metadata"][i],
            excluded_embed_metadata_keys=list(
                doc_dict["excluded_embed_metadata_keys"][i]
            ),
            excluded_llm_metadata_keys=list(doc_dict["excluded_llm_metadata_keys"][i]),
            relationships=doc_dict["relationships"][i],
            text=doc_dict["text"][i],
            mimetype=doc_dict["mimetype"][i],
        )
        documents.append(document)
    return documents


def convert_nodes_to_dict(input_nodes):
    converted_nodes = [ConvertedNode(node) for node in input_nodes]
    attributes = vars(converted_nodes[0])
    data = {key: [] for key in attributes.keys()}

    for cnode in converted_nodes:
        for key in data:
            data[key].append(getattr(cnode, key, None))
    return data
