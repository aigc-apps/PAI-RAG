from typing import Optional
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    Document,
    ImageNode,
    TextNode,
)

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

    if text is not None:
        node.set_content(text)

    return node
