# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

from langchain.vectorstores import FAISS
from langchain.vectorstores import AnalyticDB,Hologres,AlibabaCloudOpenSearch,AlibabaCloudOpenSearchSettings,ElasticsearchStore
import time
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import uuid


class myFAISS(FAISS):
    @classmethod
    def from_texts(
            cls,
            texts,
            embedding,
            metadatas=None,
            ids=None,
            values=None,
            **kwargs,
    ):
        embeddings = embedding.embed_documents(texts)
        values = texts if values is None else values
        return cls._FAISS__from(
            values,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )
    
    def add_texts(
        self,
        texts,
        metadatas=None,
        ids=None,
        values=None,
        **kwargs,
    ):
        embeddings = [self.embedding_function(k) for k in texts]
        values = texts if values is None else values
        return self._FAISS__add(values, embeddings, metadatas=metadatas, ids=ids)

class myHolo(Hologres):
    def add_texts(
        self,
        texts,
        metadatas=None,
        ids=None,
        values=None,
        **kwargs,
    ):
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        embeddings = self.embedding_function.embed_documents(list(texts))

        if not metadatas:
            metadatas = [{} for _ in texts]

        values = texts if values is None else values
        self.add_embeddings(values, embeddings, metadatas, ids, **kwargs)

        return ids
    
def getBGEReranker(model_path):
    print(f'Loading BGE Reranker from {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
    return (model, tokenizer)

class VectorDB:
    def __init__(self, args, cfg=None):
        model_dir = "/code/embedding_model"
        print('cfg[embedding][embedding_model]', cfg['embedding']['embedding_model'])
        if cfg['embedding']['embedding_model'] == "OpenAIEmbeddings":
            self.embed = OpenAIEmbeddings(openai_api_key = cfg['embedding']['openai_key'])
            emb_dim = cfg['embedding']['embedding_dimension']
        else:
            self.model_name_or_path = os.path.join(model_dir, cfg['embedding']['embedding_model'])
            self.embed = HuggingFaceEmbeddings(model_name=self.model_name_or_path,
                                            model_kwargs={'device': 'cpu'})
            emb_dim = cfg['embedding']['embedding_dimension']
        self.query_topk = cfg['query_topk']
        self.vectordb_type = args.vectordb_type

        self.bge_reranker_base = getBGEReranker(os.path.join(model_dir, "bge_reranker_base"))
        self.bge_reranker_large = getBGEReranker(os.path.join(model_dir, "bge_reranker_large"))
        
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
            PRE_DELETE = True if cfg['ADBCfg']['PRE_DELETE'] == "True" else False
            vector_db = AnalyticDB(
                embedding_function=self.embed,
                embedding_dimension=emb_dim,
                connection_string=connection_string_adb,
                collection_name=cfg['ADBCfg']['PG_COLLECTION_NAME'],
                pre_delete_collection=PRE_DELETE,
            )
            end_time = time.time()
            print("Connect AnalyticDB success. Cost time: {} s".format(end_time - start_time))
        elif self.vectordb_type == 'Hologres':
            start_time = time.time()
            connection_string_holo = myHolo.connection_string_from_db_params(
                host=cfg['HOLOCfg']['PG_HOST'],
                port=cfg['HOLOCfg']['PG_PORT'],
                database=cfg['HOLOCfg']['PG_DATABASE'],
                user=cfg['HOLOCfg']['PG_USER'],
                password=cfg['HOLOCfg']['PG_PASSWORD']
            )
            vector_db = myHolo(
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
                vector_db = myFAISS.load_local(self.faiss_path, self.embed)
            except:
                vector_db = None

        self.vectordb = vector_db

    def add_documents(self, docs):
        if not self.vectordb:
            print('add_documents faiss first')
            self.vectordb = myFAISS.from_documents(docs, self.embed)
            print('add_documents self.faiss_path', self.faiss_path)
            self.vectordb.save_local(self.faiss_path)
        else:
            if self.vectordb_type == 'FAISS':
                print('add_documents FAISS')
                self.vectordb.add_documents(docs)
                self.vectordb.save_local(self.faiss_path)
            else:
                print('add_documents else')
                self.vectordb.add_documents(docs)
    
    def add_qa_pairs(self, qa_dict, docs_dir):
        if not qa_dict:
            return
        queries = [k for k,_ in qa_dict.items()]
        answers = [v for _,v in qa_dict.items()]
        metadatas = [{
            "filename": docs_dir.rsplit('/', 1)[-1],
            "question": q
        } for q in queries]
        if not self.vectordb:
            self.vectordb = myFAISS.from_texts(queries, self.embed, metadatas=metadatas, values=answers)
            self.vectordb.save_local(self.faiss_path)
        else:
            self.vectordb.add_texts(queries, metadatas=metadatas, values=answers)
    
    def filter_docs_by_thresh(self, docs, thresh):
        return [doc for doc in docs if float(doc[1]) <= float(thresh)]

    def rerank_docs(self, query, docs, model_name):
        if model_name == "BGE-Reranker-Base":
            model, tokenizer = self.bge_reranker_base
        elif model_name == "BGE-Reranker-Large":
            model, tokenizer = self.bge_reranker_large
        
        pairs = [[query, doc[0].page_content] for doc in docs]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

        result = sorted([(doc[0], score) for doc, score in zip(docs, scores.tolist())], key=lambda x: x[1], reverse=True)
        return result

    def similarity_search_db(self, query, topk, score_threshold, model_name):
        assert self.vectordb is not None, f'error: vector db has not been set, please assign a remote type by "--vectordb_type <vectordb>" or create FAISS db by "--upload"'
        if self.vectordb_type == 'FAISS':
            self.vectordb = myFAISS.load_local(self.faiss_path, self.embed)
            # docs = self.vectordb.similarity_search_with_relevance_scores(query, k=topk,kwargs={"score_threshold": score_threshold})
            docs = self.vectordb.similarity_search_with_score(query, k=topk)
        else:
            docs = self.vectordb.similarity_search_with_score(query, k=topk)

        print('docs', docs)
        new_docs = self.filter_docs_by_thresh(docs, score_threshold)
        if model_name != 'No Re-Rank':
            new_docs = self.rerank_docs(query, new_docs, model_name)

        return new_docs
