# Copyright (c) Alibaba Cloud PAI.
# SPDX-License-Identifier: Apache-2.0
# deling.sc

from fastapi import FastAPI, File, UploadFile
import gradio as gr
from modules.LLMService import LLMService
import time
import os
from pydantic import BaseModel
import json
from args import parse_args
from modules.UI import *
import uvicorn
from utils import options
from loguru import logger

def init_args(args):
    args.config = 'configs/config_holo.json'
    args.prompt_engineering = 'simple'
    args.embed_model = "SGPT-125M-weightedmean-nli-bitfit"
    args.embed_dim = 768
    # args.vectordb_type = 'Elasticsearch'
    args.upload = False
    # args.user_query = None
    # args.query_type = "retrieval_llm"

_global_args = parse_args()
init_args(_global_args)

service = LLMService(_global_args)

with open(_global_args.config) as f:
    _global_cfg = json.load(f)
    
class Query(BaseModel):
    question: str
    topk: int | None = None
    topp: float | None = 0.8
    temperature: float | None = 0.7
    vector_topk: int | None = 3
    score_threshold: float | None = 0.5

class LLMQuery(BaseModel):
    question: str
    topk: int | None = None
    topp: float | None = 0.8
    temperature: float | None = 0.7
    
class VectorQuery(BaseModel):
    question: str
    vector_topk: int | None = 3
    score_threshold: float | None = 0.5

app = FastAPI()

# @app.post("/chat/llm")
# async def query_by_llm(query: LLMQuery):
#     ans, lens, _ = service.query_only_llm(query = query.question, llm_topK=query.topk, llm_topp=query.topp, llm_temp=query.temperature) 
#     return {"response": ans, "tokens": lens}

# @app.post("/chat/vectorstore")
# async def query_by_vectorstore(query: VectorQuery):
#     ans, lens = service.query_only_vectorstore(query = query.question, topk=query.vector_topk, score_threshold=query.score_threshold) 
#     return {"response": ans, "tokens": lens}

# @app.post("/chat/langchain")
# async def query_by_langchain(query: Query):
#     ans, lens, _ = service.query_retrieval_llm(query = query.question, topk=query.vector_topk, score_threshold=query.score_threshold, llm_topK=query.topk, llm_topp=query.topp, llm_temp=query.temperature) 
#     return {"response": ans, "tokens": lens}

# @app.post("/uploadfile")
# async def create_upload_file(file: UploadFile | None = None):
#     if not file:
#         return {"message": "No upload file sent"}
#     else:
#         fn = file.filename
#         save_path = f'./file/'
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)
    
#         save_file = os.path.join(save_path, fn)
    
#         f = open(save_file, 'wb')
#         data = await file.read()
#         f.write(data)
#         f.close()
#         service.upload_custom_knowledge(f.name,200,0)
#         return {"response": "success"}


# @app.post("/config")
# async def create_config_json_file(file: UploadFile | None = None):
#     if not file:
#         return {"message": "No upload config json file sent"}
#     else:
#         fn = file.filename
#         save_path = f'./config/'
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)
    
#         save_file = os.path.join(save_path, fn)
    
#         f = open(save_file, 'wb')
#         data = await file.read()
#         f.write(data)
#         f.close()
#         with open(f.name) as c:
#             cfg = json.load(c)
#         _global_args.embed_model = cfg['embedding']['embedding_model']
#         _global_args.vectordb_type = cfg['vector_store']
#         if 'query_topk' not in cfg:
#             cfg['query_topk'] = 4
#         if 'prompt_template' not in cfg:
#             cfg['prompt_template'] = "基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 \"根据已知信息无法回答该问题\" 或 \"没有提供足够的相关信息\"，不允许在答案中添加编造成分，答案请使用中文。\n=====\n已知信息:\n{context}\n=====\n用户问题:\n{question}"
#         if cfg.get('create_docs') is None:
#             cfg['create_docs'] = {}
#         cfg['create_docs']['chunk_size'] = 200
#         cfg['create_docs']['chunk_overlap'] = 0
#         cfg['create_docs']['docs_dir'] = 'docs/'
#         cfg['create_docs']['glob'] = "**/*"
            
#         connect_time = service.init_with_cfg(cfg,_global_args)
#         return {"response": "success"}

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

    @app.post("/chat/vectorstore")
    async def query_by_vectorstore(query: VectorQuery):
        ans, lens = service.query_only_vectorstore(query = query.question, topk=query.vector_topk, score_threshold=query.score_threshold) 
        return {"response": ans, "tokens": lens}

    @app.post("/chat/langchain")
    async def query_by_langchain(query: Query):
        ans, lens, _ = service.query_retrieval_llm(query = query.question, topk=query.vector_topk, score_threshold=query.score_threshold, llm_topK=query.topk, llm_topp=query.topp, llm_temp=query.temperature) 
        return {"response": ans, "tokens": lens}

os_env_params = {}
def get_environment_params(_global_cfg):
    os_env_params['EAS_URL'] = os.getenv('EAS_URL', 'http://127.0.0.1:8000')
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
    # app = gr.mount_gradio_app(app, ui, path='')
    app, local_url, share_url = ui.queue(
        concurrency_count=1, max_size=64
    ).launch(
        server_name="0.0.0.0",
        server_port=options.cmd_opts.port,
        prevent_thread_lock=True)
    
    logger.info("Adding fast api url...")
    add_general_url(app)
    
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level=options.cmd_opts.log_level)
    start_webui()
    while 1:
        time.sleep(0.01)
