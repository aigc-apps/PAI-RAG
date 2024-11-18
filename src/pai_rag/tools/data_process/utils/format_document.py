from llama_index.core.schema import Document


def document_to_dict(doc):
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


def dict_to_document(doc_dict):
    return Document(
        id_=doc_dict["id"],
        embedding=doc_dict["embedding"],
        metadata=doc_dict["metadata"],
        excluded_embed_metadata_keys=doc_dict["excluded_embed_metadata_keys"],
        excluded_llm_metadata_keys=doc_dict["excluded_llm_metadata_keys"],
        relationships=doc_dict["relationships"],
        text=doc_dict["text"],
        mimetype=doc_dict["mimetype"],
    )
