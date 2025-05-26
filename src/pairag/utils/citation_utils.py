import json
from llama_index.core.schema import ImageNode
from pairag.chat.models import ChatResponseWrapper


def get_citations_from_node(
    return_reference: bool, response_wrapper: ChatResponseWrapper
):
    citations = []
    citation_details = []
    if return_reference:
        for score_node in response_wrapper.source_nodes:
            if isinstance(score_node.node, ImageNode):
                url = score_node.node.image_url
                if url is not None:
                    citations.append(url)
                    citation_details.append(
                        {
                            "name": "Image",
                            "text": None,
                            "url": url,
                            "score": score_node.score,
                        }
                    )
            else:
                url = score_node.node.metadata.get(
                    "file_url"
                ) or score_node.node.metadata.get("file_path")
                citations.append(url)

                if score_node.node.metadata.get("invalid_flag") is not None:
                    citation_details.append(
                        {
                            "name": "SQL Information",
                            "text": json.dumps(
                                {
                                    "SQL": score_node.node.metadata.get(
                                        "query_code_instruction"
                                    ),
                                    "SQL_Exec_Result": score_node.node.text,
                                    "Tables": score_node.node.metadata.get(
                                        "query_tables"
                                    ),
                                    "Valid": score_node.node.metadata.get(
                                        "invalid_flag"
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                            "url": url,
                            "score": score_node.score,
                        }
                    )
                else:
                    if score_node.node.metadata.get("source") == "web_search":
                        citation_details.append(
                            {
                                "name": score_node.node.metadata.get("file_name"),
                                "text": score_node.node.text,
                                "url": url,
                                "host_name": score_node.node.metadata.get("host_name"),
                                "host_logo": score_node.node.metadata.get("host_logo"),
                                "publish_time": score_node.node.metadata.get(
                                    "publish_time"
                                ),
                                "score": score_node.score,
                            }
                        )
                    else:
                        citation_details.append(
                            {
                                "name": score_node.node.metadata.get("file_name"),
                                "text": score_node.node.text,
                                "url": url,
                                "score": score_node.score,
                            }
                        )

    return citations, citation_details
