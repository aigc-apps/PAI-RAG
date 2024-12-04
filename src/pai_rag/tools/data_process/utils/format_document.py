from llama_index.core.schema import Document


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
