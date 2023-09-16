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

def init_args(args):
    args.config = 'configs/config_holo.json'
    args.prompt_engineering = 'general'
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
    prompt: str | None = None

host_ = "127.0.0.1"
app = FastAPI(host=host_)

@app.post("/chat/llm")
async def query_by_llm(query: Query):
    ans, lens, _ = service.query_only_llm(query.question) 
    return {"response": ans, "tokens": lens}

@app.post("/chat/vectorstore")
async def query_by_vectorstore(query: Query):
    ans, lens = service.query_only_vectorstore(query.question,query.topk) 
    return {"response": ans, "tokens": lens}

@app.post("/chat/langchain")
async def query_by_langchain(query: Query):
    ans, lens, _ = service.query_retrieval_llm(query.question,query.topk,query.prompt) 
    return {"response": ans, "tokens": lens}

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile | None = None):
    if not file:
        return {"message": "No upload file sent"}
    else:
        fn = file.filename
        save_path = f'./file/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
        save_file = os.path.join(save_path, fn)
    
        f = open(save_file, 'wb')
        data = await file.read()
        f.write(data)
        f.close()
        service.upload_custom_knowledge(f.name,200,0)
        return {"response": "success"}


@app.post("/config")
async def create_config_json_file(file: UploadFile | None = None):
    if not file:
        return {"message": "No upload config json file sent"}
    else:
        fn = file.filename
        save_path = f'./config/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    
        save_file = os.path.join(save_path, fn)
    
        f = open(save_file, 'wb')
        data = await file.read()
        f.write(data)
        f.close()
        with open(f.name) as c:
            cfg = json.load(c)
        _global_args.embed_model = cfg['embedding']['embedding_model']
        _global_args.vectordb_type = cfg['vector_store']
        if 'query_topk' not in cfg:
            cfg['query_topk'] = 4
        if 'prompt_template' not in cfg:
            cfg['prompt_template'] = "基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 \"根据已知信息无法回答该问题\" 或 \"没有提供足够的相关信息\"，不允许在答案中添加编造成分，答案请使用中文。\n=====\n已知信息:\n{context}\n=====\n用户问题:\n{question}"
        if cfg.get('create_docs') is None:
            cfg['create_docs'] = {}
        cfg['create_docs']['chunk_size'] = 200
        cfg['create_docs']['chunk_overlap'] = 0
        cfg['create_docs']['docs_dir'] = 'docs/'
        cfg['create_docs']['glob'] = "**/*"
            
        connect_time = service.init_with_cfg(cfg,_global_args)
        return {"response": "success"}
    

ui = create_ui(service,_global_args,_global_cfg)
app = gr.mount_gradio_app(app, ui, path='')