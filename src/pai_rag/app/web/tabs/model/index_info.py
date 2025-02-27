from pai_rag.app.web.rag_local_client import rag_client


def get_index_map():
    index_map = rag_client.list_indexes()
    return index_map


def delete_index(index_name):
    rag_client.delete_index(index_name)
