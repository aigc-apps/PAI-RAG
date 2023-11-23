# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

from langchain.vectorstores import FAISS
from langchain.vectorstores import AnalyticDB,Hologres,AlibabaCloudOpenSearch,AlibabaCloudOpenSearchSettings,ElasticsearchStore
import time
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import os

class VectorDB:
    def __init__(self, args, cfg=None):
        model_dir = "/code/embedding_model"
        print('cfg[embedding][embedding_model]', cfg['embedding']['embedding_model'])
        if cfg['embedding']['embedding_model'] == "OpenAIEmbeddings":
            self.embed = OpenAIEmbeddings(openai_api_key = cfg['embedding']['embedding_dimension'])
            emb_dim = 1536
        else:
            self.model_name_or_path = os.path.join(model_dir, cfg['embedding']['embedding_model'])
            self.embed = HuggingFaceEmbeddings(model_name=self.model_name_or_path,
                                            model_kwargs={'device': 'cpu'})
            emb_dim = cfg['embedding']['embedding_dimension']
        self.query_topk = cfg['query_topk']
        self.vectordb_type = args.vectordb_type
        
        print('self.vectordb_type',self.vectordb_type)
        if self.vectordb_type == 'AnalyticDB':
            start_time = time.time()
            connection_string_adb = AnalyticDB.connection_string_from_db_params(
                host=cfg['ADBCfg']['PG_HOST'],
                database=cfg['ADBCfg']['PG_DATABASE'],
                user=cfg['ADBCfg']['PG_USER'],
                password=cfg['ADBCfg']['PG_PASSWORD'],
                driver='psycopg2cffi',
                port=5432,
            )
            vector_db = AnalyticDB(
                embedding_function=self.embed,
                embedding_dimension=emb_dim,
                connection_string=connection_string_adb,
                pre_delete_collection=cfg['ADBCfg']['PRE_DELETE'],
            )
            end_time = time.time()
            print("Connect AnalyticDB success. Cost time: {} s".format(end_time - start_time))
        elif self.vectordb_type == 'Hologres':
            start_time = time.time()
            connection_string_holo = Hologres.connection_string_from_db_params(
                host=cfg['HOLOCfg']['PG_HOST'],
                port=cfg['HOLOCfg']['PG_PORT'],
                database=cfg['HOLOCfg']['PG_DATABASE'],
                user=cfg['HOLOCfg']['PG_USER'],
                password=cfg['HOLOCfg']['PG_PASSWORD']
            )
            vector_db = Hologres(
                embedding_function=self.embed,
                ndims=emb_dim,
                connection_string=connection_string_holo,
                table_name=cfg['HOLOCfg']['TABLE']
            )
            end_time = time.time()
            print("Connect Hologres success. Cost time: {} s".format(end_time - start_time))
        elif self.vectordb_type == 'ElasticSearch':
            start_time = time.time()
            vector_db = ElasticsearchStore(
                 es_url=cfg['ElasticSearchCfg']['ES_URL'],
                 index_name=cfg['ElasticSearchCfg']['ES_INDEX'],
                 es_user=cfg['ElasticSearchCfg']['ES_USER'],
                 es_password=cfg['ElasticSearchCfg']['ES_PASSWORD'],
                 embedding=self.embed
            )
            end_time = time.time()
            print("Connect ElasticSearchStore success. Cost time: {} s".format(end_time - start_time))
        elif self.vectordb_type == 'OpenSearch':
            start_time = time.time()
            print("Start Connect AlibabaCloudOpenSearch ")
            settings = AlibabaCloudOpenSearchSettings(
                endpoint=cfg['OpenSearchCfg']['endpoint'],
                instance_id=cfg['OpenSearchCfg']['instance_id'],
                datasource_name=cfg['OpenSearchCfg']['datasource_name'],
                username=cfg['OpenSearchCfg']['username'],
                password=cfg['OpenSearchCfg']['password'],
                embedding_index_name=cfg['OpenSearchCfg']['embedding_index_name'],
                field_name_mapping={
                    "id": cfg['OpenSearchCfg']['field_name_mapping']['id'],
                    "document": cfg['OpenSearchCfg']['field_name_mapping']['document'],
                    "embedding": cfg['OpenSearchCfg']['field_name_mapping']['embedding'],
                    "source": cfg['OpenSearchCfg']['field_name_mapping']['source'],
                },
            )
            vector_db = AlibabaCloudOpenSearch(
                embedding=self.embed, config=settings
            )
            end_time = time.time()
            print("Connect AlibabaCloudOpenSearch success. Cost time: {} s".format(end_time - start_time))
        elif self.vectordb_type == 'FAISS':
            print("Not config any database, use FAISS-cpu default.")
            vector_db = None
            if not os.path.exists(cfg['FAISS']['index_path']):
                os.makedirs(cfg['FAISS']['index_path'])
                print('已创建目录：', cfg['FAISS']['index_path'])
            else:
                print('目录已存在：', cfg['FAISS']['index_path'])
            self.faiss_path = os.path.join(cfg['FAISS']['index_path'],cfg['FAISS']['index_name'])
            try:
                vector_db = FAISS.load_local(self.faiss_path, self.embed)
            except:
                vector_db = None

        self.vectordb = vector_db

    def add_documents(self, docs):
        if not self.vectordb:
            print('add_documents faiss')
            self.vectordb = FAISS.from_documents(docs, self.embed)
            self.vectordb.save_local(self.faiss_path)
        else:
            print('add_documents else')
            self.vectordb.add_documents(docs)

    def similarity_search_db(self, query, topk):
        assert self.vectordb is not None, f'error: vector db has not been set, please assign a remote type by "--vectordb_type <vectordb>" or create FAISS db by "--upload"'
        return self.vectordb.similarity_search(query, k=topk)
