import gradio as gr
from modules.LLMService import LLMService
import time
import os
import json
  
def create_ui(service,_global_args,_global_cfg, host_, port_):
    url_str = 'http://' + host_ + ':' + str(port_)
    api_url = url_str + '/docs'
    
    def connect_adb(emb_model, emb_dim, eas_url, eas_token, pg_host, pg_user, pg_pwd, pg_database, pg_del):
        cfg = {
            'embedding': {
                "embedding_model": emb_model,
                "model_dir": "./embedding_model/",
                "embedding_dimension": emb_dim
            },
            'EASCfg': {
                "url": eas_url,
                "token": eas_token
            },
            'ADBCfg': {
                "PG_HOST": pg_host,
                "PG_DATABASE": pg_database,
                "PG_USER": pg_user,
                "PG_PASSWORD": pg_pwd,
                "PRE_DELETE": pg_del
            }
        }
        _global_args.vectordb_type = "adb"
        _global_cfg.update(cfg)
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect AnalyticDB success."

    def connect_holo(emb_model, emb_dim, eas_url, eas_token, pg_host, pg_database, pg_user, pg_pwd):
        cfg = {
            'embedding': {
                "embedding_model": emb_model,
                "model_dir": "./embedding_model/",
                "embedding_dimension": emb_dim
            },
            'EASCfg': {
                "url": eas_url,
                "token": eas_token
            },
            'HOLOCfg': {
                "PG_HOST": pg_host,
                "PG_DATABASE": pg_database,
                "PG_PORT": 80,
                "PG_USER": pg_user,
                "PG_PASSWORD": pg_pwd
            }
        }
        _global_args.vectordb_type = "holo"
        _global_cfg.update(cfg)
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect Hologres success."

    def connect_es(emb_model, emb_dim, eas_url, eas_token, es_url, es_index, es_user, es_pwd):
        cfg = {
            'embedding': {
                "embedding_model": emb_model,
                "model_dir": "./embedding_model/",
                "embedding_dimension": emb_dim
            },
            'EASCfg': {
                "url": eas_url,
                "token": eas_token
            },
            'ElasticSearchCfg': {
                "ES_URL": es_url,
                "ES_INDEX": es_index,
                "ES_USER": es_user,
                "ES_PASSWORD": es_pwd
            }
        }
        _global_args.vectordb_type = "elastic_search"
        _global_cfg.update(cfg)
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect ElasticSearch success."
    
    def connect_faiss(emb_model, emb_dim, eas_url, eas_token, path, name):
        cfg = {
            "embedding": {
                "model_dir": "./embedding_model/",
                "embedding_model": emb_model,
                "embedding_dimension": emb_dim
            },

            "EASCfg": {
                "url": eas_url,
                "token": eas_token
            },
            
            "FAISS": {
                "index_path": path,
                "index_name": name
            }
        }
        _global_args.vectordb_type = "faiss"
        _global_cfg.update(cfg)
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect FAISS success."
    
    with gr.Blocks() as demo:
 
        value_md =  """
            # Chatbot Langchain with LLM on PAI ! 

            Build your own personalized knowledge base question-answering chatbot. 
            
            Referenced API: """ + api_url + """
            
            - Platform of Artificial Intelligence: https://help.aliyun.com/zh/pai/       
            
            - PAI-EAS : https://www.aliyun.com/product/bigdata/learn/eas
            
            - Supported VectorStore:  [Hologres](https://www.aliyun.com/product/bigdata/hologram)  /  [Elasticsearch](https://www.aliyun.com/product/bigdata/elasticsearch)  /  [AnalyticDB](https://www.aliyun.com/product/apsaradb/gpdb)  /  [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss) (only used for testing)
                
            """
            
        gr.Markdown(value=value_md)
                
        with gr.Tab("Settings"):
            with gr.Row():
                with gr.Column():
                    config_file = gr.File(label="Upload a local config json file",
                                    file_types=['.json'], file_count="single", interactive=True)
                    cfg_btn = gr.Button("Parse config json")
          
                    with gr.Column(label='Emebdding Config'):
                        emb_model = gr.Dropdown(["SGPT-125M-weightedmean-nli-bitfit", "text2vec-large-chinese","text2vec-base-chinese", "paraphrase-multilingual-MiniLM-L12-v2"], label="Emebdding Model", value=_global_args.embed_model)
                        emb_dim = gr.Textbox(label="Emebdding Dimension", value=_global_args.embed_dim)
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
                        eas_url = gr.Textbox(label="EAS Url", value=_global_cfg['EASCfg']['url'])
                        eas_token = gr.Textbox(label="EAS Token", value=_global_cfg['EASCfg']['token'])
                    
                with gr.Column():
                    vs_radio = gr.Dropdown(
                        [ "Hologres", "ElasticSearch", "AnalyticDB", "FAISS"], label="Which VectorStore do you want to use?")
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
                chunk_size = gr.Textbox(label="Chunk Size (The size of the chunks into which a document is divided)",value='200')
                chunk_overlap = gr.Textbox(label="Chunk Overlap (The portion of adjacent document chunks that overlap with each other)",value='0')
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
                        service.upload_custom_knowledge(file_path,int(chunk_size),int(chunk_overlap))
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
                with gr.Column(label="Prompt Design"):
                    prm_radio = gr.Radio(
                        [ "General", "Extract URL", "Accurate Content", "Customize"], label="Please choose the prompt template type"
                    )
                    prompt = gr.Textbox(placeholders="Please choose the prompt template", lines=4)
                    def change_prompt_template(prm_radio):
                        if prm_radio == "General":
                            return {prompt: gr.update(value="基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 \"根据已知信息无法回答该问题\" 或 \"没有提供足够的相关信息\"，不允许在答案中添加编造成分，答案请使用中文。\n=====\n已知信息:\n{context}\n=====\n用户问题:\n{question}")}
                        elif prm_radio == "Extract URL":
                            return {prompt: gr.update(value="你是一位智能小助手，请根据下面我所提供的相关知识，对我提出的问题进行回答。回答的内容必须包括其定义、特征、应用领域以及相关网页链接等等内容，同时务必满足下方所提的要求！\n=====\n 知识库相关知识如下:\n{context}\n=====\n 请根据上方所提供的知识库内容与要求，回答以下问题:\n {question}")}
                        elif prm_radio == "Accurate Content":
                            return {prompt: gr.update(value="你是一位知识小助手，请根据下面我提供的知识库中相关知识，对我提出的若干问题进行回答，同时回答的内容需满足我所提的要求! \n=====\n 知识库相关知识如下:\n{context}\n=====\n 请根据上方所提供的知识库内容与要求，回答以下问题:\n {question}")}
                        elif prm_radio == "Customize":
                            return {prompt: gr.update(value="")}
                    prm_radio.change(fn=change_prompt_template, inputs=prm_radio, outputs=[prompt])
            
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Enter your question.")
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                clear = gr.ClearButton([msg, chatbot])

            print('topk.value', topk.value)
            print('prompt.value', prompt.value)
            def respond(message, chat_history, ds_radio, topk, prm_radio, prompt):
                if ds_radio == "Vector Store":
                    answer = service.query_only_vectorstore(message,topk)
                elif ds_radio == "LLM":
                    answer = service.query_only_llm(message)         
                else:
                    answer = service.query_retrieval_llm(message,topk, prm_radio, prompt)
                bot_message = answer
                chat_history.append((message, bot_message))
                time.sleep(2)
                return "", chat_history

            submitBtn.click(respond, [msg, chatbot, ds_radio, topk, prm_radio, prompt], [msg, chatbot])
    return demo