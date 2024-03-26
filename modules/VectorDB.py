# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

from langchain.vectorstores import FAISS
from langchain.vectorstores import AnalyticDB,Hologres,AlibabaCloudOpenSearch,AlibabaCloudOpenSearchSettings,ElasticsearchStore,Milvus
import time
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import uuid
import jieba
import logging
import json
import torch
from loguru import logger
from utils.load_utils import *

CACHE_DB_FILE = 'cache/db_file.jsonl'

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
        values = values or texts
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
        embeddings = self._embed_documents(texts)
        values = values or texts
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

class myElasticSearch(ElasticsearchStore):
    def add_texts(
        self,
        texts,
        metadatas=None,
        ids=None,
        values=None,
        refresh_indices=True,
        create_index_if_not_exists=True,
        **kwargs,
    ):
        try:
            from elasticsearch.helpers import BulkIndexError, bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        embeddings = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        requests = []

        values = texts if values is None else values
        if self.embedding is not None:
            # If no search_type requires inference, we use the provided
            # embedding function to embed the texts.
            embeddings = self.embedding.embed_documents(list(texts))
            dims_length = len(embeddings[0])

            if create_index_if_not_exists:
                self._create_index_if_not_exists(
                    index_name=self.index_name, dims_length=dims_length
                )

            for i, (text, vector) in enumerate(zip(values, embeddings)):
                metadata = metadatas[i] if metadatas else {}

                requests.append(
                    {
                        "_op_type": "index",
                        "_index": self.index_name,
                        self.query_field: text,
                        self.vector_query_field: vector,
                        "metadata": metadata,
                        "_id": ids[i],
                    }
                )

        else:
            # the search_type doesn't require inference, so we don't need to
            # embed the texts.
            if create_index_if_not_exists:
                self._create_index_if_not_exists(index_name=self.index_name)

            for i, text in enumerate(values):
                metadata = metadatas[i] if metadatas else {}

                requests.append(
                    {
                        "_op_type": "index",
                        "_index": self.index_name,
                        self.query_field: text,
                        "metadata": metadata,
                        "_id": ids[i],
                    }
                )

        if len(requests) > 0:
            try:
                success, failed = bulk(
                    self.client, requests, stats_only=True, refresh=refresh_indices
                )
                logger.debug(
                    f"Added {success} and failed to add {failed} texts to index"
                )

                logger.debug(f"added texts {ids} to index")
                return ids
            except BulkIndexError as e:
                logger.error(f"Error adding texts: {e}")
                firstError = e.errors[0].get("index", {}).get("error", {})
                logger.error(f"First error reason: {firstError.get('reason')}")
                raise e

        else:
            logger.debug("No texts to add to index")
            return []

def chinese_text_preprocess_func(text: str):
    return [t for t in jieba.cut(text) if t != ' ']

def getBGEReranker(model_path):
    logger.info(f'Loading BGE Reranker from {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
    return (model, tokenizer)

class VectorDB:
    weights = [0.5, 0.5]
    """ Weights of ensembled retrievers for Reciprocal Rank Fusion."""

    def __init__(self, args, cfg=None):
        self.model_dir = "/huggingface/sentence_transformers"
        logger.info(f"Using embedding_model: {cfg['embedding']['embedding_model']}")
        if cfg['embedding']['embedding_model'] == "OpenAIEmbeddings":
            self.embed = OpenAIEmbeddings(openai_api_key = cfg['embedding']['openai_key'])
            self.emb_dim = cfg['embedding']['embedding_dimension']
        else:
            self.model_name_or_path = os.path.join(self.model_dir, cfg['embedding']['embedding_model'])
            self.embed = HuggingFaceEmbeddings(model_name=self.model_name_or_path,
                                            model_kwargs={'device': 'cuda:0'})
            self.emb_dim = cfg['embedding']['embedding_dimension']
        self.query_topk = cfg['query_topk']
        self.vectordb_type = args.vectordb_type
        self.bm25_load_cache = args.bm25_load_cache
        self.is_rerank = False
        
        cache_contents, cache_metadatas = self.load_cache(contents=[], metadatas=[])
        if len(cache_contents)>0:
            self.bm25_retriever = BM25Retriever.from_texts(cache_contents, metadatas=cache_metadatas, preprocess_func=chinese_text_preprocess_func)
    
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
                embedding_dimension=self.emb_dim,
                connection_string=connection_string_adb,
                collection_name=cfg['ADBCfg']['PG_COLLECTION_NAME'],
                pre_delete_collection=PRE_DELETE,
            )
            end_time = time.time()
            logger.info("Connect AnalyticDB success. Cost time: {} s".format(end_time - start_time))
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
                ndims=self.emb_dim,
                connection_string=connection_string_holo,
                table_name=cfg['HOLOCfg']['TABLE']
            )
            end_time = time.time()
            logger.info("Connect Hologres success. Cost time: {} s".format(end_time - start_time))
        elif self.vectordb_type == 'ElasticSearch':
            start_time = time.time()
            vector_db = myElasticSearch(
                 es_url=cfg['ElasticSearchCfg']['ES_URL'],
                 index_name=cfg['ElasticSearchCfg']['ES_INDEX'],
                 es_user=cfg['ElasticSearchCfg']['ES_USER'],
                 es_password=cfg['ElasticSearchCfg']['ES_PASSWORD'],
                 embedding=self.embed
            )
            end_time = time.time()
            logger.info("Connect ElasticSearchStore success. Cost time: {} s".format(end_time - start_time))
        elif self.vectordb_type == 'OpenSearch':
            start_time = time.time()
            logger.info("Start Connect AlibabaCloudOpenSearch ")
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
            logger.info("Connect AlibabaCloudOpenSearch success. Cost time: {} s".format(end_time - start_time))
        elif self.vectordb_type == 'FAISS':
            logger.info("Not config any database, use FAISS-cpu default.")
            vector_db = None
            if not os.path.exists(cfg['FAISS']['index_path']):
                os.makedirs(cfg['FAISS']['index_path'])
                logger.info('已创建目录：', cfg['FAISS']['index_path'])
            else:
                logger.info('目录已存在：', cfg['FAISS']['index_path'])
            self.faiss_path = os.path.join(cfg['FAISS']['index_path'],cfg['FAISS']['index_name'])
            try:
                vector_db = myFAISS.load_local(self.faiss_path, self.embed)
            except:
                vector_db = None
        elif self.vectordb_type == 'Milvus':
            logger.info("Start connect Milvus")
            start_time = time.time()
            DROP_OLD = True if cfg['MilvusCfg']['DROP'] == "True" else False
            vector_db = Milvus(
                embedding_function=self.embed,
                collection_name=cfg['MilvusCfg']['COLLECTION'],
                metadata_field="meta",
                connection_args={
                    "host": cfg['MilvusCfg']['HOST'],
                    "port": cfg['MilvusCfg']['PORT'],
                    "user": cfg['MilvusCfg']['USER'],
                    "password": cfg['MilvusCfg']['PASSWORD']
                },
                drop_old=DROP_OLD
            )
            end_time = time.time()
            logger.info("Connect Milvus success. Cost time: {} s".format(end_time - start_time))

        self.vectordb = vector_db
        
    def update_cache(self, contents, metadatas):
        with open(CACHE_DB_FILE, 'a+') as f:
            for c, m in zip(contents, metadatas):
                f.write(json.dumps({
                    "page_content": c,
                    "metadata": m
                }) + '\n')
    
    def load_cache(self, contents, metadatas):
        # load cache data
        cache_data = []
        if os.path.exists(CACHE_DB_FILE):
            if not self.bm25_load_cache:
                os.remove(CACHE_DB_FILE)
            else:
                with open(CACHE_DB_FILE, 'r') as f:
                    cache_data = [json.loads(line) for line in f.readlines()]
        cache_contents = [line['page_content'] for line in cache_data]
        cache_metadatas = [line['metadata'] for line in cache_data]
        logger.info(f"[INFO] cache doc num: {len(cache_contents)}")

        # save new data
        self.update_cache(contents, metadatas)

        # merge cache data and new data
        return contents+cache_contents, metadatas+cache_metadatas

    def add_documents(self, docs):
        if not self.vectordb:
            logger.info('add_documents faiss first')
            self.vectordb = myFAISS.from_documents(docs, self.embed)
            logger.info('add_documents self.faiss_path', self.faiss_path)
            self.vectordb.save_local(self.faiss_path)
        else:
            if self.vectordb_type == 'FAISS':
                logger.info('add_documents to FAISS')
                self.vectordb.add_documents(docs)
                self.vectordb.save_local(self.faiss_path)
            else:
                logger.info('add_documents other vectordb')
                self.vectordb.add_documents(docs)
        
        new_contents = [doc.page_content for doc in docs]
        new_metadatas = [doc.metadata for doc in docs]
        logger.info(f"[INFO] new doc num: {len(new_contents)}")
        contents, metadatas = self.load_cache(new_contents, new_metadatas)
        logger.info(f"[INFO] final doc num: {len(contents)}")

        # self.bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=chinese_text_preprocess_func)
        self.bm25_retriever = BM25Retriever.from_texts(contents, metadatas=metadatas, preprocess_func=chinese_text_preprocess_func)
    
    def add_qa_pairs(self, qa_dict, docs_dir):
        if not qa_dict:
            return
        queries = list(qa_dict.keys())
        answers = list(qa_dict.values())
        metadatas = [{
            "filename": docs_dir.rsplit('/', 1)[-1],
            "question": q
        } for q in queries]
        if not self.vectordb:
            self.vectordb = myFAISS.from_texts(queries, self.embed, metadatas=metadatas, values=answers)
        else:
            self.vectordb.add_texts(queries, metadatas=metadatas, values=answers)
        if self.vectordb_type == 'FAISS':
            self.vectordb.save_local(self.faiss_path)
        
        contents, metadatas = self.load_cache(answers, metadatas)
        # qa_texts = [f"{q} {a}" for q,a in zip(queries, answers)]
        self.bm25_retriever = BM25Retriever.from_texts(contents, metadatas=metadatas, preprocess_func=chinese_text_preprocess_func)
    
    def filter_docs_by_thresh(self, docs, thresh):
        return [doc for doc in docs if float(doc[1]) <= float(thresh)]

    def rerank_docs(self, query, docs, model_name):
        if not self.is_rerank:
            logger.info("Loading bge-reranker-base and bge-reranker-large for the first time.")
            self.bge_reranker_base = getBGEReranker(os.path.join(self.model_dir, "bge-reranker-base"))
            self.bge_reranker_large = getBGEReranker(os.path.join(self.model_dir, "bge-reranker-large"))
            self.is_rerank = True
        
        if model_name == "BGE-Reranker-Base":    
            model, tokenizer = self.bge_reranker_base
        elif model_name == "BGE-Reranker-Large":
            model, tokenizer = self.bge_reranker_large
        
        docs_list = [item[0] if isinstance(item, tuple) else item for item in docs]
        pairs = [[query, doc.page_content] for doc in docs_list]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

        result = sorted([(doc, score) for doc, score in zip(docs_list, scores.tolist())], key=lambda x: x[1], reverse=True)
        return result

    def similarity_search_db(self, query, topk, score_threshold, model_name, kw_retrieval):
        assert self.vectordb is not None, f'error: vector db has not been set, please assign a remote type by "--vectordb_type <vectordb>" or create FAISS db by "--upload"'
        if self.vectordb_type == 'FAISS':
            self.vectordb = myFAISS.load_local(self.faiss_path, self.embed)
            # docs = self.vectordb.similarity_search_with_relevance_scores(query, k=topk,kwargs={"score_threshold": score_threshold})
        if kw_retrieval != 'Embedding Only':
            logger.info(f"[INFO] Using Both Embedding Retrieval and BM25 Retrieval")
            self.bm25_retriever.k = topk
            self.embed_retriever = self.vectordb.as_retriever(search_kwargs={"k": topk})
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.embed_retriever], weights=self.weights
            )
            docs = self.ensemble_retriever.get_relevant_documents(query)
        else:
            logger.info(f"[INFO] Using Embedding Retrieval ONLY")
            docs = self.vectordb.similarity_search_with_score(query, k=topk)
            docs = self.filter_docs_by_thresh(docs, score_threshold)

        logger.info('[INFO] docs:\n', docs)
        if model_name != 'No Re-Rank':
            docs = self.rerank_docs(query, docs, model_name)

        if len(docs) > topk:
            docs = docs[:topk]
            
        return docs
