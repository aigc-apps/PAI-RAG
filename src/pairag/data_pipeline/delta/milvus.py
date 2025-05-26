from typing import Dict
from pymilvus import Collection

from pairag.data_pipeline.constants import (
    DEFAULT_MODIFIED_AT_FIELD,
    DEFAULT_NODE_SOURCE_FIELD,
)
from pairag.data_pipeline.delta.models import DocItem


def list_docs_in_milvus_collection(
    collection: Collection,
    oss_path_prefix: str,
) -> Dict[str, DocItem]:
    # only return the documents under the oss_path_prefix
    if oss_path_prefix:
        expr = f"{DEFAULT_NODE_SOURCE_FIELD} like '{oss_path_prefix}%'"
    else:
        expr = None

    iterator = collection.query_iterator(
        batch_size=10,
        output_fields=[DEFAULT_NODE_SOURCE_FIELD, DEFAULT_MODIFIED_AT_FIELD],
        expr=expr,
    )
    results: Dict[str, DocItem] = {}
    while True:
        fetch_data = iterator.next()
        if not fetch_data:
            iterator.close()
            break

        # scan the fetch data, get the latest modified time and the node ids
        for record in fetch_data:
            source = record[DEFAULT_NODE_SOURCE_FIELD]
            if source not in results:
                results[source] = DocItem(
                    doc_path=source,
                    modified_time=record[DEFAULT_MODIFIED_AT_FIELD],
                    node_ids=[record["id"]],
                )
            else:
                results[source].modified_time = min(
                    record[DEFAULT_MODIFIED_AT_FIELD],
                    results[source].modified_time,
                )
                results[source].node_ids.append(record["id"])

    return results
