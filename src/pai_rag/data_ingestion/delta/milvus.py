from typing import Dict, Optional
from pymilvus import Collection
from pymilvus import connections as milvus_connections
from langstudio.rag.index_manifest import IndexManifest

from pai_rag.data_ingestion.constants import DEFAULT_MODIFIED_AT_FIELD, DEFAULT_NODE_SOURCE_FIELD
from pai_rag.data_ingestion.delta.models import DocItem


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

def list_docs_in_milvus_from_langstudio_index_manifest(
    index_manifest: IndexManifest, oss_path_prefix: Optional[str] = None
) -> Dict[str, DocItem]:
    """
    List the documents in the vector database.

    Args:
        index_manifest (IndexManifest): The index manifest.
        oss_path_prefix (Optional[str]): The prefix of the OSS path, used to filter the documents
            under the prefix.

    Returns:
        Dict[str, DocumentVectorStoreEntry]: The documents in the vector database, the
            key is the OSS path of the document, the value is a tuple of the latest
            modified time and the node ids.
    """
    connection = index_manifest.vector_store_connection
    milvus_connections.connect(
        uri=connection.uri,
        token=connection.token,
    )
    collection = Collection(index_manifest.store.collection_name)

    return list_docs_in_milvus_collection(
        collection=collection,
        oss_path_prefix=oss_path_prefix
    )