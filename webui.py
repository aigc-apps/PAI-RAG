# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

from fastapi import FastAPI
from modules.LLMService import LLMService
import time
import os
from pydantic import BaseModel
import json
from args import parse_args
from modules.UI import *
import sys
from loguru import logger

# Disable unstructured analytics tracking to mitigate timeout issues during PAI-EAS service.
# This is achieved by setting an environment variable that the unstructured library recognizes.
# By doing this, we prevent the library's internal function `scarf_analytics()` from making
# network requests to "https://packages.unstructured.io", which was causing timeouts.

os.environ["SCARF_NO_ANALYTICS"] = "true"


_global_args = parse_args()
service = LLMService()

with open(_global_args.config) as f:
    _global_cfg = json.load(f)

class Query(BaseModel):
    question: str
    topk: int | None = None
    topp: float | None = 0.8
    temperature: float | None = 0.7
    vector_topk: int | None = 3
    score_threshold: float | None = 600
    rerank_model: str | None = 'No Re-Rank'
    kw_retrieval: str | None = 'Embedding Only'

class LLMQuery(BaseModel):
    question: str
    topk: int | None = None
    topp: float | None = 0.8
    temperature: float | None = 0.7

class VectorQuery(BaseModel):
    question: str
    vector_topk: int | None = 3
    score_threshold: float | None = 600
    rerank_model: str | None = 'No Re-Rank'
    kw_retrieval: str | None = 'Embedding Only'

app = FastAPI()

def setup_middleware(app):
    # reset current middleware to allow modifying user provided list
    app.middleware_stack = None
    configure_cors_middleware(app)
    app.build_middleware_stack()  # rebuild middleware stack on-the-fly

def configure_cors_middleware(app):
    from fastapi.middleware.cors import CORSMiddleware

    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_credentials": True,
    }

    app.add_middleware(CORSMiddleware, **cors_options)

def add_general_url(
    app
):    
    setup_middleware(app)
    @app.post("/chat/llm")
    async def query_by_llm(query: LLMQuery):
        ans, lens, _ = service.query_only_llm(query = query.question, llm_topK=query.topk, llm_topp=query.topp, llm_temp=query.temperature) 
        return {"response": ans, "tokens": lens}

    @app.post("/chat/retrieval")
    async def query_by_vectorstore(query: VectorQuery):
        ans, lens = service.query_only_vectorstore(query = query.question, topk=query.vector_topk, score_threshold=query.score_threshold, rerank_model=query.rerank_model, kw_retrieval=query.kw_retrieval) 
        return {"response": ans, "tokens": lens}

    @app.post("/chat/rag")
    async def query_by_langchain(query: Query):
        ans, lens, _ = service.query_retrieval_llm(query = query.question, topk=query.vector_topk, score_threshold=query.score_threshold, llm_topK=query.topk, llm_topp=query.topp, llm_temp=query.temperature, rerank_model=query.rerank_model, kw_retrieval=query.kw_retrieval) 
        return {"response": ans, "tokens": lens}

os_env_params = {}
def get_environment_params(_global_cfg):
    os_env_params['EAS_URL'] = os.getenv('EAS_URL', 'http://127.0.0.1:8000')
    os_env_params['EAS_TOKEN'] = os.getenv('EAS_TOKEN', '')
    os_env_params['VECTOR_STORE'] = os.getenv('VECTOR_STORE', 'FAISS')
    os_env_params['FAISS_PATH'] = os.getenv('FAISS_PATH', '/code')
    os_env_params['FAISS_INDEX'] = os.getenv('FAISS_INDEX', 'faiss')
    
    os_env_params['ADB_PG_HOST'] = os.getenv('ADB_PG_HOST', '')
    os_env_params['ADB_PG_DATABASE'] = os.getenv('ADB_PG_DATABASE', 'postgres')
    os_env_params['ADB_PG_USER'] = os.getenv('ADB_PG_USER', '')
    os_env_params['ADB_PG_PASSWORD'] = os.getenv('ADB_PG_PASSWORD', '')
    os_env_params['ADB_PG_COLLECTION_NAME'] = os.getenv('ADB_PG_COLLECTION_NAME', '')
    os_env_params['ADB_PRE_DELETE'] = os.getenv('ADB_PRE_DELETE', "False")
    
    os_env_params['HOLO_HOST'] = os.getenv('HOLO_HOST', '')
    os_env_params['HOLO_DATABASE'] = os.getenv('HOLO_DATABASE', '')
    os_env_params['HOLO_USER'] = os.getenv('HOLO_USER', '')
    os_env_params['HOLO_PASSWORD'] = os.getenv('HOLO_PASSWORD', '')
    os_env_params['HOLO_TABLE'] = os.getenv('HOLO_TABLE', '')
    
    os_env_params['ES_URL'] = os.getenv('ES_URL', '')
    os_env_params['ES_INDEX'] = os.getenv('ES_INDEX', '')
    os_env_params['ES_USER'] = os.getenv('ES_USER', '')
    os_env_params['ES_PASSWORD'] = os.getenv('ES_PASSWORD', '')
    
    os_env_params['MILVUS_COLLECTION'] = os.getenv('MILVUS_COLLECTION', '')
    os_env_params['MILVUS_HOST'] = os.getenv('MILVUS_HOST', '')
    os_env_params['MILVUS_PORT'] = os.getenv('MILVUS_PORT', '')
    os_env_params['MILVUS_USER'] = os.getenv('MILVUS_USER', '')
    os_env_params['MILVUS_PASSWORD'] = os.getenv('MILVUS_PASSWORD', '')
    os_env_params['MILVUS_DROP'] = os.getenv('MILVUS_DROP', "False")
    
    _global_cfg['vector_store'] = os_env_params['VECTOR_STORE']
    if _global_cfg['vector_store'] == "FAISS":
        _global_cfg['FAISS']['index_path'] = os_env_params['FAISS_PATH']
        _global_cfg['FAISS']['index_name'] = os_env_params['FAISS_INDEX']
    elif _global_cfg['vector_store'] == "AnalyticDB":
        _global_cfg['ADBCfg'] = {}
        _global_cfg['ADBCfg']['PG_HOST'] = os_env_params['ADB_PG_HOST']
        _global_cfg['ADBCfg']['PG_DATABASE'] = os_env_params['ADB_PG_DATABASE']
        _global_cfg['ADBCfg']['PG_USER'] = os_env_params['ADB_PG_USER']
        _global_cfg['ADBCfg']['PG_PASSWORD'] = os_env_params['ADB_PG_PASSWORD']
        _global_cfg['ADBCfg']['PG_COLLECTION_NAME'] = os_env_params['ADB_PG_COLLECTION_NAME']
        _global_cfg['ADBCfg']['PRE_DELETE'] = os_env_params['ADB_PRE_DELETE']
    elif _global_cfg['vector_store'] == "Hologres":
        _global_cfg['HOLOCfg']['PG_HOST'] = os_env_params['HOLO_HOST']
        _global_cfg['HOLOCfg']['PG_DATABASE'] = os_env_params['HOLO_DATABASE']
        _global_cfg['HOLOCfg']['PG_USER'] = os_env_params['HOLO_USER']
        _global_cfg['HOLOCfg']['PG_PASSWORD'] = os_env_params['HOLO_PASSWORD']
        _global_cfg['HOLOCfg']['TABLE'] = os_env_params['HOLO_TABLE']
    elif _global_cfg['vector_store'] == "ElasticSearch":
        _global_cfg['ElasticSearchCfg'] = {}
        _global_cfg['ElasticSearchCfg']['ES_URL'] = os_env_params['ES_URL']
        _global_cfg['ElasticSearchCfg']['ES_INDEX'] = os_env_params['ES_INDEX']
        _global_cfg['ElasticSearchCfg']['ES_USER'] = os_env_params['ES_USER']
        _global_cfg['ElasticSearchCfg']['ES_PASSWORD'] = os_env_params['ES_PASSWORD']
    elif _global_cfg['vector_store'] == "Milvus":
        _global_cfg['MilvusCfg'] = {}
        _global_cfg['MilvusCfg']['COLLECTION'] = os_env_params['MILVUS_COLLECTION']
        _global_cfg['MilvusCfg']['HOST'] = os_env_params['MILVUS_HOST']
        _global_cfg['MilvusCfg']['PORT'] = os_env_params['MILVUS_PORT']
        _global_cfg['MilvusCfg']['USER'] = os_env_params['MILVUS_USER']
        _global_cfg['MilvusCfg']['PASSWORD'] = os_env_params['MILVUS_PASSWORD']
        _global_cfg['MilvusCfg']['DROP'] = os_env_params['MILVUS_DROP']

def start_webui():
    global app
    get_environment_params(_global_cfg)
    
    logger.info("Starting Webui server...")
    ui = create_ui(service, _global_args, _global_cfg, os_env_params)
    app, _, _ = ui.queue(
        concurrency_count=1, max_size=64
    ).launch(
        server_name="0.0.0.0",
        server_port=_global_args.port,
        prevent_thread_lock=True,
        # required in local env
        share=True
        )
    
    logger.info("Adding fast api url...")
    add_general_url(app)

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level=_global_args.log_level)
    start_webui()
    while 1:
        time.sleep(0.01)
