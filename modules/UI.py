import gradio as gr
from modules.LLMService import LLMService
import time
import os
import json
import sys
import gradio

def html_path(filename):
    script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(script_path, "html", filename)

def html(filename):
    path = html_path(filename)
    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()

    return ""

def webpath(fn):
    script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)
    return f'file={web_path}?{os.path.getmtime(fn)}'

def css_html():
    head = ""
    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'
    
    cssfile = "style.css"
    if not os.path.isfile(cssfile):
        print("cssfile not exist")

    head += stylesheet(cssfile)

    return head

def reload_javascript():
    css = css_html()
    GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response
    
def create_ui(service,_global_args,_global_cfg):
    reload_javascript()
    
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
            },
            "create_docs":{
                "chunk_size": 200,
                "chunk_overlap": 0,
                "docs_dir": "docs/",
                "glob": "**/*"
            }
        }
        _global_args.vectordb_type = "AnalyticDB"
        _global_cfg.update(cfg)
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect AnalyticDB success."

    def connect_holo(emb_model, emb_dim, eas_url, eas_token, pg_host, pg_database, pg_user, pg_pwd, table):
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
                "PG_PASSWORD": pg_pwd,
                "TABLE": table
            },
            "create_docs":{
                "chunk_size": 200,
                "chunk_overlap": 0,
                "docs_dir": "docs/",
                "glob": "**/*"
            }
        }
        _global_args.vectordb_type = "Hologres"
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
            },
            "create_docs":{
                "chunk_size": 200,
                "chunk_overlap": 0,
                "docs_dir": "docs/",
                "glob": "**/*"
            }
        }
        _global_args.vectordb_type = "ElasticSearch"
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
            },
            "create_docs":{
                "chunk_size": 200,
                "chunk_overlap": 0,
                "docs_dir": "docs/",
                "glob": "**/*"
            }
        }
        _global_args.vectordb_type = "FAISS"
        _global_cfg.update(cfg)
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect FAISS success."
    
    with gr.Blocks() as demo:
 
        value_md =  """
            #  <center> \N{fire} Chatbot Langchain with LLM on PAI ! 

            ### <center> \N{rocket} Build your own personalized knowledge base question-answering chatbot. 
                        
            <center> 
            
            \N{fire} Platform: [PAI](https://help.aliyun.com/zh/pai)  /  [PAI-EAS](https://www.aliyun.com/product/bigdata/learn/eas)  / [PAI-DSW](https://pai.console.aliyun.com/notebook)
            
            \N{rocket} Supported VectorStores:  [Hologres](https://www.aliyun.com/product/bigdata/hologram)  /  [ElasticSearch](https://www.aliyun.com/product/bigdata/elasticsearch)  /  [AnalyticDB](https://www.aliyun.com/product/apsaradb/gpdb)  /  [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss)
                
            """
        
        
        gr.Markdown(value=value_md)
        api_hl = ("<div style='text-align: center;'> \N{whale} <a href='/docs'>Referenced API</a>    \N{rocket} <a href='https://github.com/aigc-apps/LLM_Solution.git'> Github Code</a>  </div>")
        gr.HTML(api_hl,elem_id='api')
                
        with gr.Tab("\N{rocket} Settings"):
            with gr.Row():
                with gr.Column():
                    with gr.Column():
                        md_emb = gr.Markdown(value="**Please set your embedding model.**")
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
                    
                    with gr.Column():
                        md_eas = gr.Markdown(value="**Please set your EAS LLM.**")
                        eas_url = gr.Textbox(label="EAS Url", value=_global_cfg['EASCfg']['url'])
                        eas_token = gr.Textbox(label="EAS Token", value=_global_cfg['EASCfg']['token'])
                    
                    with gr.Column():
                      md_cfg = gr.Markdown(value="**(Optional) Please upload your config file.**")
                      config_file = gr.File(value=_global_args.config,label="Upload a local config json file",file_types=['.json'], file_count="single", interactive=True)
                      cfg_btn = gr.Button("Parse Config", variant="primary")
                    
                with gr.Column():
                    md_vs = gr.Markdown(value="**Please set your Vector Store.**")
                    vs_radio = gr.Dropdown(
                        [ "Hologres", "ElasticSearch", "AnalyticDB", "FAISS"], label="Which VectorStore do you want to use?", value = _global_cfg['vector_store'])
                    with gr.Column(visible=(_global_cfg['vector_store']=="AnalyticDB")) as adb_col:
                        pg_host = gr.Textbox(label="Host", 
                                             value=_global_cfg['ADBCfg']['PG_HOST'] if _global_cfg['vector_store']=="AnalyticDB" else '')
                        pg_user = gr.Textbox(label="User", 
                                             value=_global_cfg['ADBCfg']['PG_USER'] if _global_cfg['vector_store']=="AnalyticDB" else '')
                        pg_database = gr.Textbox(label="Database", 
                                                 value='postgres' if _global_cfg['vector_store']=="AnalyticDB" else '')
                        pg_pwd= gr.Textbox(label="Password", 
                                           value=_global_cfg['ADBCfg']['PG_PASSWORD'] if _global_cfg['vector_store']=="AnalyticDB" else '')
                        pg_del = gr.Dropdown(["True","False"], label="Pre Delete", value=_global_cfg['ADBCfg']['PRE_DELETE'] if _global_cfg['vector_store']=="AnalyticDB" else '')
                        # pg_del = gr.Textbox(label="Pre_delete", 
                        #                     value="False" if _global_cfg['vector_store']=="AnalyticDB" else '')
                        connect_btn = gr.Button("Connect AnalyticDB", variant="primary")
                        con_state = gr.Textbox(label="Connection Info: ")
                        connect_btn.click(fn=connect_adb, inputs=[emb_model, emb_dim, eas_url, eas_token, pg_host, pg_user, pg_pwd, pg_database, pg_del], outputs=con_state, api_name="connect_adb")   
                    with gr.Column(visible=(_global_cfg['vector_store']=="Hologres")) as holo_col:
                        holo_host = gr.Textbox(label="Host",
                                               value=_global_cfg['HOLOCfg']['PG_HOST'] if _global_cfg['vector_store']=="Hologres" else '')
                        holo_database = gr.Textbox(label="Database",
                                                   value=_global_cfg['HOLOCfg']['PG_DATABASE'] if _global_cfg['vector_store']=="Hologres" else '')
                        holo_user = gr.Textbox(label="User",
                                               value=_global_cfg['HOLOCfg']['PG_USER'] if _global_cfg['vector_store']=="Hologres" else '')
                        holo_pwd= gr.Textbox(label="Password",
                                             value=_global_cfg['HOLOCfg']['PG_PASSWORD'] if _global_cfg['vector_store']=="Hologres" else '')
                        holo_table= gr.Textbox(label="Table",
                                             value=_global_cfg['HOLOCfg']['TABLE'] if _global_cfg['vector_store']=="Hologres" else '')
                        connect_btn = gr.Button("Connect Hologres", variant="primary")
                        con_state = gr.Textbox(label="Connection Info: ")
                        connect_btn.click(fn=connect_holo, inputs=[emb_model, emb_dim, eas_url, eas_token, holo_host, holo_database, holo_user, holo_pwd, holo_table], outputs=con_state, api_name="connect_holo") 
                    with gr.Column(visible=(_global_cfg['vector_store']=="ElasticSearch")) as es_col:
                        es_url = gr.Textbox(label="URL",
                                            value=_global_cfg['ElasticSearchCfg']['ES_URL'] if _global_cfg['vector_store']=="ElasticSearch" else '')
                        es_index = gr.Textbox(label="Index",
                                              value=_global_cfg['ElasticSearchCfg']['ES_INDEX'] if _global_cfg['vector_store']=="ElasticSearch" else '')
                        es_user = gr.Textbox(label="User",
                                             value=_global_cfg['ElasticSearchCfg']['ES_USER'] if _global_cfg['vector_store']=="ElasticSearch" else '')
                        es_pwd= gr.Textbox(label="Password",
                                           value=_global_cfg['ElasticSearchCfg']['ES_PASSWORD'] if _global_cfg['vector_store']=="ElasticSearch" else '')
                        connect_btn = gr.Button("Connect ElasticSearch", variant="primary")
                        con_state = gr.Textbox(label="Connection Info: ")
                        connect_btn.click(fn=connect_es, inputs=[emb_model, emb_dim, eas_url, eas_token, es_url, es_index, es_user, es_pwd], outputs=con_state, api_name="connect_es") 
                    with gr.Column(visible=(_global_cfg['vector_store']=="FAISS")) as faiss_col:
                        faiss_path = gr.Textbox(label="Path", 
                                                value = _global_cfg['FAISS']['index_path'] if _global_cfg['vector_store']=="FAISS" else '')
                        faiss_name = gr.Textbox(label="Index", 
                                                value=_global_cfg['FAISS']['index_name'] if _global_cfg['vector_store']=="FAISS" else '')
                        connect_btn = gr.Button("Connect Faiss", variant="primary")
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
                    vs_radio.change(fn=change_ds_conn, inputs=vs_radio, outputs=[adb_col,holo_col,es_col,faiss_col])
            with gr.Row():
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
                            pg_del: gr.update(value=cfg['ADBCfg']['PRE_DELETE'] if ( 'PRE_DELETE' in cfg['ADBCfg']) else 'False'),
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
                            holo_table: gr.update(value=cfg['HOLOCfg']['TABLE']),
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
                cfg_btn.click(fn=cfg_analyze, inputs=config_file, outputs=[emb_model,emb_dim,eas_url,eas_token,vs_radio,pg_host,pg_user,pg_pwd,pg_database, pg_del, holo_host, holo_database, holo_user, holo_pwd, holo_table, es_url, es_index, es_user, es_pwd, faiss_path, faiss_name], api_name="cfg_analyze")   
                
        with gr.Tab("\N{whale} Upload"):
            with gr.Row():
                with gr.Column(scale=2):
                    chunk_size = gr.Textbox(label="\N{rocket} Chunk Size (The size of the chunks into which a document is divided)",value='200')
                    chunk_overlap = gr.Textbox(label="\N{fire} Chunk Overlap (The portion of adjacent document chunks that overlap with each other)",value='0')
                with gr.Column(scale=8):
                    with gr.Tab("Files"):
                        upload_file = gr.File(label="Upload a knowledge file (supported type: txt, md, doc, docx, pdf)",
                                        file_types=['.txt', '.md', '.docx', '.pdf'], file_count="multiple")
                        connect_btn = gr.Button("Upload", variant="primary")
                        state_hl_file = gr.Textbox(label="Upload State")
                        
                    with gr.Tab("Directory"):
                        upload_file_dir = gr.File(label="Upload a knowledge directory (supported type: txt, md, docx, pdf)" , file_count="directory")
                        connect_dir_btn = gr.Button("Upload", variant="primary")
                        state_hl_dir = gr.Textbox(label="Upload State")

                    
                    def upload_knowledge(upload_file,chunk_size,chunk_overlap):
                        file_name = ''
                        for file in upload_file:
                            if file.name.lower().endswith(".txt") or file.name.lower().endswith(".md") or file.name.lower().endswith(".docx") or file.name.lower().endswith(".doc") or file.name.lower().endswith(".pdf"):
                                file_path = file.name
                                file_name += file.name.rsplit('/', 1)[-1] + ', '
                                service.upload_custom_knowledge(file_path,int(chunk_size),int(chunk_overlap))
                        return "Upload " + str(len(upload_file)) + " files [ " +  file_name + "] Success! \n \n Relevant content has been added to the vector store, you can now start chatting and asking questions." 
                    
                    def upload_knowledge_dir(upload_dir,chunk_size,chunk_overlap):
                        for file in upload_dir:
                            if file.name.lower().endswith(".txt") or file.name.lower().endswith(".md") or file.name.lower().endswith(".docx") or file.name.lower().endswith(".doc") or file.name.lower().endswith(".pdf"):
                                file_path = file.name
                                service.upload_custom_knowledge(file_path,chunk_size,chunk_overlap)
                        return "Directory: Upload " + str(len(upload_dir)) + " files Success!" 

                    connect_btn.click(fn=upload_knowledge, inputs=[upload_file,chunk_size,chunk_overlap], outputs=state_hl_file, api_name="upload_knowledge")
                    connect_dir_btn.click(fn=upload_knowledge_dir, inputs=[upload_file_dir,chunk_size,chunk_overlap], outputs=state_hl_dir, api_name="upload_knowledge_dir")
        
        with gr.Tab("\N{fire} Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    ds_radio = gr.Radio(
                        [ "Vector Store", "LLM", "Vector Store + LLM"], label="\N{fire} Which query do you want to use?"
                    )
                    topk = gr.Textbox(label="Retrieval top K answers",value='3')
                    with gr.Column():
                        prm_radio = gr.Radio(
                            [ "General", "Extract URL", "Accurate Content", "Customize"], label="\N{rocket} Please choose the prompt template type"
                        )
                        prompt = gr.Textbox(label="prompt template", placeholder="This is a prompt template", lines=4)
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
                        cur_tokens = gr.Textbox(label="\N{fire} Current total count of tokens")
                with gr.Column(scale=8):
                    chatbot = gr.Chatbot(height=500)
                    msg = gr.Textbox(label="Enter your question.")
                    with gr.Row():
                        submitBtn = gr.Button("Submit", variant="primary")
                        summaryBtn = gr.Button("Summary", variant="primary")
                        clear_his = gr.Button("Clear History", variant="secondary")
                        clear = gr.ClearButton([msg, chatbot])
                   
                    def respond(message, chat_history, ds_radio, topk, prm_radio, prompt):
                        summary_res = ""
                        if ds_radio == "Vector Store":
                            answer, lens = service.query_only_vectorstore(message,topk)
                        elif ds_radio == "LLM":
                            answer, lens, summary_res = service.query_only_llm(message)         
                        else:
                            answer, lens, summary_res = service.query_retrieval_llm(message,topk, prm_radio, prompt)
                        bot_message = answer
                        chat_history.append((message, bot_message))
                        time.sleep(0.05)
                        return "", chat_history, str(lens) + "\n" + summary_res

                    def clear_hisoty(chat_history):
                        chat_history = []
                        service.langchain_chat_history = []
                        service.input_tokens = []
                        # chat_history.append(('Clear the chat history', bot_message))
                        time.sleep(0.05)
                        return chat_history, "0 \n Clear history successfully!"
                    
                    def summary_hisoty(chat_history):
                        service.input_tokens = []
                        bot_message = service.checkout_history_and_summary(summary=True)
                        chat_history.append(('请对我们之前的对话内容进行总结。', bot_message))
                        tokens_len = service.sp.encode(service.input_tokens, out_type=str)
                        lens = sum(len(tl) for tl in tokens_len)
                        time.sleep(0.05)
                        return chat_history, str(lens) + "\n" + bot_message
                    
                    submitBtn.click(respond, [msg, chatbot, ds_radio, topk, prm_radio, prompt], [msg, chatbot, cur_tokens])
                    clear_his.click(clear_hisoty,[chatbot],[chatbot, cur_tokens])
                    summaryBtn.click(summary_hisoty,[chatbot],[chatbot, cur_tokens])
    
        footer = html("footer.html")
        gr.HTML(footer, elem_id="footer")
        
    return demo