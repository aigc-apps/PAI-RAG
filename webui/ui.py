from fastapi import FastAPI, File, UploadFile
import gradio as gr
from webui import *
from LLMService import LLMService
import random
import time
import os
from loguru import logger
from pydantic import BaseModel
import json
from langchain.document_loaders import UnstructuredFileLoader

  
def create_ui(service):
    def connect_adb(emb_model, emb_dim, eas_url, eas_token, pg_host, pg_user, pg_pwd, pg_database, pg_del):
        cfg = {
            'embedding': {},
            'EASCfg': {},
            'ADBCfg': {}
        }
        cfg['embedding']['embedding_model'] = emb_model
        cfg['embedding']['model_dir'] = "/code/embedding_model/"
        cfg['embedding']['embedding_dimension'] = emb_dim
        cfg['EASCfg']['url'] = eas_url
        cfg['EASCfg']['token'] = eas_token
        cfg['ADBCfg']['PG_HOST'] = pg_host
        cfg['ADBCfg']['PG_DATABASE'] = pg_database
        cfg['ADBCfg']['PG_USER'] = pg_user
        cfg['ADBCfg']['PG_PASSWORD'] = pg_pwd
        cfg['ADBCfg']['PG_DATABASE'] = pg_database
        cfg['ADBCfg']['PRE_DELETE'] = pg_del
        connect_time = service.init_with_cfg(cfg)
        return "Connect AnalyticDB success. Cost time: {} s".format(connect_time)

    def connect_holo(emb_model, emb_dim, eas_url, eas_token, pg_host, pg_database, pg_user, pg_pwd):
        cfg = {
            'embedding': {},
            'EASCfg': {},
            'HOLOCfg': {}
        }
        cfg['embedding']['embedding_model'] = emb_model
        cfg['embedding']['model_dir'] = "/code/embedding_model/"
        cfg['embedding']['embedding_dimension'] = emb_dim
        cfg['EASCfg']['url'] = eas_url
        cfg['EASCfg']['token'] = eas_token
        cfg['HOLOCfg']['PG_HOST'] = pg_host
        cfg['HOLOCfg']['PG_DATABASE'] = pg_database
        cfg['HOLOCfg']['PG_PORT'] = 80
        cfg['HOLOCfg']['PG_USER'] = pg_user
        cfg['HOLOCfg']['PG_PASSWORD'] = pg_pwd
        connect_time = service.init_with_cfg(cfg)
        return "Connect Hologres success. Cost time: {} s".format(connect_time)

    def connect_es(emb_model, emb_dim, eas_url, eas_token, es_url, es_index, es_user, es_pwd):
        cfg = {
            'embedding': {},
            'EASCfg': {},
            'ElasticSearchCfg': {}
        }
        cfg['embedding']['embedding_model'] = emb_model
        cfg['embedding']['model_dir'] = "/code/embedding_model/"
        cfg['embedding']['embedding_dimension'] = emb_dim
        cfg['EASCfg']['url'] = eas_url
        cfg['EASCfg']['token'] = eas_token
        cfg['ElasticSearchCfg']['ES_URL'] = es_url
        cfg['ElasticSearchCfg']['ES_INDEX'] = es_index
        cfg['ElasticSearchCfg']['ES_USER'] = es_user
        cfg['ElasticSearchCfg']['ES_PASSWORD'] = es_pwd
        connect_time = service.init_with_cfg(cfg)
        return "Connect ElasticSearch success. Cost time: {} s".format(connect_time)
    
    def connect_faiss(emb_model, emb_dim, eas_url, eas_token, path, name):
        cfg = {
            'embedding': {},
            'EASCfg': {},
            'FAISS': {}
        }
        cfg['embedding']['embedding_model'] = emb_model
        cfg['embedding']['model_dir'] = "/code/embedding_model/"
        cfg['embedding']['embedding_dimension'] = emb_dim
        cfg['EASCfg']['url'] = eas_url
        cfg['EASCfg']['token'] = eas_token
        cfg['FAISS']['index_path'] = path
        cfg['FAISS']['index_name'] = name
        connect_time = service.init_with_cfg(cfg)
        return "Connect FAISS success. Cost time: {} s".format(connect_time)
    
    with gr.Blocks() as demo:
        with gr.Tab("Settings"):
            with gr.Row():
                with gr.Column():
                    config_file = gr.File(label="Upload a local config json file",
                                    file_types=['.json'], file_count="single", interactive=True)
                    cfg_btn = gr.Button("Parse config json")
          
                    with gr.Column(label='Emebdding Config'):
                        emb_model = gr.Dropdown(["SGPT-125M-weightedmean-nli-bitfit", "text2vec-large-chinese","text2vec-base-chinese", "paraphrase-multilingual-MiniLM-L12-v2"], label="Emebdding Model")
                        emb_dim = gr.Textbox(label="Emebdding Dimension")
                        def change_emb_model(model):
                            if model == "SGPT-125M-weightedmean-nli-bitfit":
                                return {emb_dim: gr.update(value="768")}
                            if model == "text2vec-large-chinese":
                                return {emb_dim: gr.update(value="1024")}
                            if model == "text2vec-base-chinese":
                                return {emb_dim: gr.update(value="768")}
                            if model == "paraphrase-multilingual-MiniLM-L12-v2":
                                return {emb_dim: gr.update(value="384")}
                        emb_model.change(fn=change_emb_model, inputs=emb_model, outputs=[emb_dim])
                    
                    with gr.Column(label='EAS Config'):
                        eas_url = gr.Textbox(label="EAS Url")
                        eas_token = gr.Textbox(label="EAS Token")
                    
                with gr.Column():
                    vs_radio = gr.Dropdown(
                        [ "Hologres", "ElasticSearch", "AnalyticDB", "FAISS"], label="Which VectorStore do you want to use?"
                    )
                    with gr.Column(visible=False) as adb_col:
                        pg_host = gr.Textbox(label="Host")
                        pg_user = gr.Textbox(label="User")
                        pg_database = gr.Textbox(label="Database",default='postgres')
                        pg_pwd= gr.Textbox(label="Password")
                        pg_del = gr.Textbox(label="Pre_delete")
                        connect_btn = gr.Button("Connect AnalyticDB")
                        con_state = gr.Textbox(label="Connection Info: ")
                        connect_btn.click(fn=connect_adb, inputs=[emb_model, emb_dim, eas_url, eas_token, pg_host, pg_user, pg_pwd, pg_database, pg_del], outputs=con_state, api_name="connect_adb")   
                    with gr.Column(visible=False) as holo_col:
                        holo_host = gr.Textbox(label="Host")
                        holo_database = gr.Textbox(label="Database")
                        holo_user = gr.Textbox(label="User")
                        holo_pwd= gr.Textbox(label="Password")
                        connect_btn = gr.Button("Connect Hologres")
                        con_state = gr.Textbox(label="Connection Info: ")
                        connect_btn.click(fn=connect_holo, inputs=[emb_model, emb_dim, eas_url, eas_token, holo_host, holo_database, holo_user, holo_pwd], outputs=con_state, api_name="connect_holo") 
                    with gr.Column(visible=False) as es_col:
                        es_url = gr.Textbox(label="URL")
                        es_index = gr.Textbox(label="Index")
                        es_user = gr.Textbox(label="User")
                        es_pwd= gr.Textbox(label="Password")
                        connect_btn = gr.Button("Connect ElasticSearch")
                        con_state = gr.Textbox(label="Connection Info: ")
                        connect_btn.click(fn=connect_es, inputs=[emb_model, emb_dim, eas_url, eas_token, es_url, es_index, es_user, es_pwd], outputs=con_state, api_name="connect_es") 
                    with gr.Column(visible=False) as faiss_col:
                        faiss_path = gr.Textbox(label="Path")
                        faiss_name = gr.Textbox(label="Index")
                        connect_btn = gr.Button("Connect Faiss")
                        con_state = gr.Textbox(label="Connection Info: ")
                        connect_btn.click(fn=connect_faiss, inputs=[emb_model, emb_dim, eas_url, eas_token, faiss_path, faiss_name], outputs=con_state, api_name="connect_faiss") 
                    def change_ds_conn(radio):
                        if radio=="AnalyticDB":
                            return {adb_col: gr.update(visible=True), holo_col: gr.update(visible=False), es_col: gr.update(visible=False), faiss_col: gr.update(visible=False)}
                        elif radio=="Hologres":
                            return {adb_col: gr.update(visible=False), holo_col: gr.update(visible=True), es_col: gr.update(visible=False), faiss_col: gr.update(visible=False)}
                        elif radio=="ElasticSearch":
                            return {adb_col: gr.update(visible=False), holo_col: gr.update(visible=False), es_col: gr.update(visible=True), faiss_col: gr.update(visible=False)}
                        elif radio=="FAISS":
                            return {adb_col: gr.update(visible=False), holo_col: gr.update(visible=False), es_col: gr.update(visible=False), faiss_col: gr.update(visible=True)}
            with gr.Row():
                vs_radio.change(fn=change_ds_conn, inputs=vs_radio, outputs=[adb_col,holo_col,es_col,faiss_col])
                
                def cfg_analyze(config_file):
                    filepath = config_file.name
                    with open(filepath) as f:
                        cfg = json.load(f)
                    if cfg['vector_store'] == "AnalyticDB":
                        return {
                            emb_model: gr.update(value=cfg['embedding']['embedding_model']), 
                            emb_dim: gr.update(value=cfg['embedding']['embedding_dimension']),
                            eas_url: gr.update(value=cfg['EASCfg']['url']), 
                            eas_token:  gr.update(value=cfg['EASCfg']['token']),
                            vs_radio: gr.update(value=cfg['vector_store']),
                            pg_host: gr.update(value=cfg['ADBCfg']['PG_HOST']),
                            pg_user: gr.update(value=cfg['ADBCfg']['PG_USER']),
                            pg_pwd: gr.update(value=cfg['ADBCfg']['PG_PASSWORD']),
                            pg_database: gr.update(value=cfg['ADBCfg']['PG_DATABASE'] if ( 'PG_DATABASE' in cfg['ADBCfg']) else 'postgres'),
                            pg_del: gr.update(value=cfg['ADBCfg']['pre_delete'] if ( 'pre_delete' in cfg['ADBCfg']) else 'False'),
                            }
                    if cfg['vector_store'] == "Hologres":
                        return {
                            emb_model: gr.update(value=cfg['embedding']['embedding_model']), 
                            emb_dim: gr.update(value=cfg['embedding']['embedding_dimension']),
                            eas_url: gr.update(value=cfg['EASCfg']['url']), 
                            eas_token:  gr.update(value=cfg['EASCfg']['token']),
                            vs_radio: gr.update(value=cfg['vector_store']),
                            holo_host: gr.update(value=cfg['HOLOCfg']['PG_HOST']),
                            holo_database: gr.update(value=cfg['HOLOCfg']['PG_DATABASE']),
                            holo_user: gr.update(value=cfg['HOLOCfg']['PG_USER']),
                            holo_pwd: gr.update(value=cfg['HOLOCfg']['PG_PASSWORD']),
                            }
                    if cfg['vector_store'] == "ElasticSearch":
                        return {
                            emb_model: gr.update(value=cfg['embedding']['embedding_model']), 
                            emb_dim: gr.update(value=cfg['embedding']['embedding_dimension']),
                            eas_url: gr.update(value=cfg['EASCfg']['url']), 
                            eas_token:  gr.update(value=cfg['EASCfg']['token']),
                            vs_radio: gr.update(value=cfg['vector_store']),
                            es_url: gr.update(value=cfg['ElasticSearchCfg']['ES_URL']),
                            es_index: gr.update(value=cfg['ElasticSearchCfg']['ES_INDEX']),
                            es_user: gr.update(value=cfg['ElasticSearchCfg']['ES_USER']),
                            es_pwd: gr.update(value=cfg['ElasticSearchCfg']['ES_PASSWORD']),
                            }
                    if cfg['vector_store'] == "FAISS":
                        return {
                            emb_model: gr.update(value=cfg['embedding']['embedding_model']), 
                            emb_dim: gr.update(value=cfg['embedding']['embedding_dimension']),
                            eas_url: gr.update(value=cfg['EASCfg']['url']), 
                            eas_token:  gr.update(value=cfg['EASCfg']['token']),
                            vs_radio: gr.update(value=cfg['vector_store']),
                            faiss_path: gr.update(value=cfg['FAISS']['index_path']),
                            faiss_name: gr.update(value=cfg['FAISS']['index_name'])
                            }
                cfg_btn.click(fn=cfg_analyze, inputs=config_file, outputs=[emb_model,emb_dim,eas_url,eas_token,vs_radio,pg_host,pg_user,pg_pwd,pg_database, pg_del, holo_host, holo_database, holo_user, holo_pwd, es_url, es_index, es_user, es_pwd, faiss_path, faiss_name], api_name="cfg_analyze")   
                
        with gr.Tab("Upload"):
            with gr.Row():
                chunk_size = gr.Textbox(label="Chunk Size (The size of the chunks into which a document is divided)")
                chunk_overlap = gr.Textbox(label="Chunk Overlap (The portion of adjacent document chunks that overlap with each other)")
            with gr.Tab("Files"):
                upload_file = gr.File(label="Upload a knowledge file (supported type: txt, md, doc, docx, pdf)",
                                file_types=['.txt', '.md', '.docx', '.pdf'], file_count="multiple")
                connect_btn = gr.Button("Upload")
                state_hl_file = gr.Textbox(label="Upload State")
                
            with gr.Tab("Directory"):
                upload_file_dir = gr.File(label="Upload a knowledge directory (supported type: txt, md, docx, pdf)" , file_types=['text'], file_count="directory")
                connect_dir_btn = gr.Button("Upload")
                state_hl_dir = gr.Textbox(label="Upload State")

            
            def upload_knowledge(upload_file,chunk_size,chunk_overlap):
                for file in upload_file:
                    if file.name.lower().endswith(".txt") or file.name.lower().endswith(".md") or file.name.lower().endswith(".docx") or file.name.lower().endswith(".doc") or file.name.lower().endswith(".pdf"):
                        file_path = file.name
                        service.upload_custom_knowledge(file_path,chunk_size,chunk_overlap)
                return "File: Upload " + str(len(upload_file)) + " files Success!" 
            
            def upload_knowledge_dir(upload_dir,chunk_size,chunk_overlap):
                for file in upload_dir:
                    if file.name.lower().endswith(".txt") or file.name.lower().endswith(".md") or file.name.lower().endswith(".docx") or file.name.lower().endswith(".doc") or file.name.lower().endswith(".pdf"):
                        file_path = file.name
                        service.upload_custom_knowledge(file_path,chunk_size,chunk_overlap)
                return "Directory: Upload " + str(len(upload_dir)) + " files Success!" 

            connect_btn.click(fn=upload_knowledge, inputs=[upload_file,chunk_size,chunk_overlap], outputs=state_hl_file, api_name="upload_knowledge")
            connect_dir_btn.click(fn=upload_knowledge_dir, inputs=[upload_file_dir,chunk_size,chunk_overlap], outputs=state_hl_dir, api_name="upload_knowledge_dir")
        
        with gr.Tab("Chat"):
            ds_radio = gr.Radio(
                [ "Vector Store", "LLM", "Vector Store + LLM"], label="Which query do you want to use?"
            )
            with gr.Row():
                topk = gr.Textbox(label="Retrieval top K answers",value='3')
                prompt = gr.Textbox(label="Prompt Design", value="基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 \"根据已知信息无法回答该问题\" 或 \"没有提供足够的相关信息\"，不允许在答案中添加编造成分，答案请使用中文。\n已知信息:{context}\n用户问题:{question}", lines=4)
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Enter your question.")
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                clear = gr.ClearButton([msg, chatbot])

            print('topk.value', topk.value)
            print('prompt.value', prompt.value)
            def respond(message, chat_history, ds_radio, topk, prompt):
                if ds_radio == "Vector Store":
                    answer = service.query_only_vectorestore(message,topk)
                elif ds_radio == "LLM":
                    answer = service.query_only_llm(message)         
                else:
                    answer = service.user_query(message,topk,prompt)
                bot_message = answer
                chat_history.append((message, bot_message))
                time.sleep(2)
                return "", chat_history

            submitBtn.click(respond, [msg, chatbot, ds_radio, topk, prompt], [msg, chatbot])
    return demo