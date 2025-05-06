import json
from typing import Any, Dict, Optional
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    Document,
    ImageNode,
    TextNode,
    MetadataMode,
)
from pai_rag.data_ingestion.constants import (
    DEFAULT_MD5_FIELD,
    DEFAULT_MODIFIED_AT_FIELD,
    DEFAULT_NODE_SOURCE_FIELD,
)

PERSIST_COLUMN_FIELDS = [
    "id",
    "text",
    "doc_id",
    "document_id",
    "file_name",
    "file_path",
    "file_type",
    "last_modified_date",
    "ref_doc_id",
    "embedding",
    "sparse_embedding",
    "excluded_embed_metadata_keys",
    "excluded_llm_metadata_keys",
    "operation",
    "operation_reason",
    "_node_content",
    "_node_type",
    DEFAULT_MD5_FIELD,
    DEFAULT_MODIFIED_AT_FIELD,
    DEFAULT_NODE_SOURCE_FIELD,
]


def metadata_dict_to_node_v2(metadata: dict, text: Optional[str] = None) -> BaseNode:
    """Common logic for loading Node data from metadata dict."""
    node_json = metadata.get("_node_content", None)
    node_type = metadata.get("_node_type", None)
    if node_json is None:
        raise ValueError("Node content not found in metadata dict.")

    node: BaseNode
    if node_type == IndexNode.class_name():
        node = IndexNode.from_json(node_json)
    elif node_type == ImageNode.class_name():
        node = ImageNode.from_json(node_json)
    elif node_type == Document.class_name():
        node = Document.from_json(node_json)
    else:
        node = TextNode.from_json(node_json)

    if node.embedding is None:
        node.embedding = metadata.get("embedding")

    if text is not None:
        node.set_content(text)

    return node


def node_to_metadata_dict_v2(
    node: BaseNode,
) -> Dict[str, Any]:
    """Common logic for saving Node data into metadata dict."""
    # Using mode="json" here because BaseNode may have fields of type bytes (e.g. images in ImageBlock),
    # which would cause serialization issues.
    node_dict = node.model_dump(mode="json")
    metadata: Dict[str, Any] = node_dict.get("metadata", {})

    # dump remainder of node_dict to json string
    metadata["_node_content"] = json.dumps(node_dict, ensure_ascii=False)
    metadata["_node_type"] = node.class_name()

    # store ref doc id at top level to allow metadata filtering
    # kept for backwards compatibility, will consolidate in future
    metadata["document_id"] = node.ref_doc_id or "None"  # for Chroma
    metadata["doc_id"] = node.ref_doc_id or "None"  # for Pinecone, Qdrant, Redis
    metadata["ref_doc_id"] = node.ref_doc_id or "None"  # for Weaviate
    metadata["id"] = node.id_
    metadata["text"] = node.get_content(metadata_mode=MetadataMode.NONE)

    metadata_with_schema = {k: metadata.get(k) for k in PERSIST_COLUMN_FIELDS}
    return metadata_with_schema
