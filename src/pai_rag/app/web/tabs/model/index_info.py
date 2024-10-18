from pai_rag.app.web.rag_client import rag_client


def get_index_map():
    index_map = rag_client.list_indexes()
    return index_map
