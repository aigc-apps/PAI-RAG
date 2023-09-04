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
port_ = 8098
app = FastAPI(host=host_,port=port_)

@app.post("/chat/llm")
async def query_by_llm(query: Query):
    ans = service.query_only_llm(query.question) 
    return {"response": ans}

@app.post("/chat/vectorstore")
async def query_by_vectorstore(query: Query):
    ans = service.query_only_vectorstore(query.question,query.topk) 
    return {"response": ans}

@app.post("/chat/langchain")
async def query_by_langchain(query: Query):
    ans = service.user_query(query.question,query.topk,query.prompt) 
    return {"response": ans}

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
        connect_time = service.init_with_cfg(cfg)
        return {"response": "success"}
    

ui = create_ui(service,_global_args,_global_cfg, host_, port_)
app = gr.mount_gradio_app(app, ui, path='')
