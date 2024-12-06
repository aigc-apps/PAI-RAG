from llama_index.core.schema import Document
from datetime import datetime
from llama_index.core.schema import TextNode, ImageNode
import numpy as np


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
        data["excluded_embed_metadata_keys"].append(
            np.array(node.excluded_embed_metadata_keys)
        ),
        data["excluded_llm_metadata_keys"].append(
            np.array(node.excluded_llm_metadata_keys)
        ),
        # image
        data["image"].append(node.image if type(node) is ImageNode else None),
        data["image_path"].append(node.image_path if type(node) is ImageNode else None),
        data["image_url"].append(node.image_url if type(node) is ImageNode else None),
        data["image_mimetype"].append(
            node.image_mimetype if type(node) is ImageNode else None
        )
    return data
