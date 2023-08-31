import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    def __init__(self, model_name):
        model_dir = "embedding_model"
        model_name_or_path = os.path.join(model_dir, model_name)
        self.embed = HuggingFaceEmbeddings(model_name=model_name_or_path,
                                           model_kwargs={'device': 'cpu'})

    def embed_query(self, query):
        return self.embed.embed_query(query)
