# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

import os
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    def __init__(self, model_name):
        model_dir = "embedding_model"
        self.model_name_or_path = os.path.join(model_dir, model_name)
        self.embed = HuggingFaceEmbeddings(model_name=self.model_name_or_path,
                                           model_kwargs={'device': 'cpu'})

    # def embed_query(self, query):
    #     return self.embed.embed_query(query)

    # def embed_documents(self, query):
    #     return self.embed.embed_documents(query)
