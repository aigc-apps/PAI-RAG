import gradio as gr
import time
import os
import json
from loguru import logger

CACHE_DIR = "cache/"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
CACHE_CONFIG_NAME = "config.json"

def html(filename):
    script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.join(script_path, "html", filename)
    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()

    return ""

def create_ui(service, _global_args, _global_cfg, os_env_params):
    eas_url = os_env_params["EAS_URL"]
    eas_token = os_env_params["EAS_TOKEN"]
    llm_src = "EAS"
    open_api_key = ""

    def get_llm_cfg(llm_src, eas_url, eas_token, open_api_key):
        if llm_src == "EAS":
            cfg = {"LLM": "EAS", "EASCfg": {"url": eas_url, "token": eas_token}}
        elif llm_src == "OpenAI":
            cfg = {"LLM": "OpenAI", "OpenAI": {"key": open_api_key}}
        return cfg

    def check_db_cache(keys, new_config):
        cache_path = os.path.join(CACHE_DIR, CACHE_CONFIG_NAME)
        # Check if there is local cache for bm25
        if not os.path.exists(cache_path):
            with open(cache_path, "w+") as f:
                json.dump(new_config, f)
            return False

        # Read cached config file
        with open(cache_path, "r") as f:
            cache_config = json.load(f)
        # Check if new_config is consistent with cache_config
        res = all([cache_config[k] == new_config[k] for k in keys])

        # Update cached config file
        with open(cache_path, "w+") as f:
            json.dump(new_config, f)

        return res

    def connect_adb(
        emb_model,
        emb_dim,
        emb_openai_key,
        pg_host,
        pg_user,
        pg_pwd,
        pg_database,
        pg_collection,
        pg_del,
    ):
        cfg = get_llm_cfg(llm_src, eas_url, eas_token, open_api_key)
        cfg_db = {
            "embedding": {
                "embedding_model": emb_model,
                "model_dir": "./embedding_model/",
                "embedding_dimension": emb_dim,
                "openai_key": emb_openai_key,
            },
            "ADBCfg": {
                "PG_HOST": pg_host,
                "PG_DATABASE": pg_database,
                "PG_USER": pg_user,
                "PG_PASSWORD": pg_pwd,
                "PG_COLLECTION_NAME": pg_collection,
                "PRE_DELETE": pg_del,
            },
            "create_docs": {
                "chunk_size": 200,
                "chunk_overlap": 0,
                "docs_dir": "docs/",
                "glob": "**/*",
            },
        }
        cfg.update(cfg_db)
        _global_args.vectordb_type = "AnalyticDB"
        _global_cfg.update(cfg)
        _global_args.bm25_load_cache = check_db_cache(
            ["vector_store", "ADBCfg"], _global_cfg
        )
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect AnalyticDB success."

    def connect_holo(
        emb_model, emb_dim, emb_openai_key, pg_host, pg_database, pg_user, pg_pwd, table
    ):
        cfg = get_llm_cfg(llm_src, eas_url, eas_token, open_api_key)
        cfg_db = {
            "embedding": {
                "embedding_model": emb_model,
                "model_dir": "./embedding_model/",
                "embedding_dimension": emb_dim,
                "openai_key": emb_openai_key,
            },
            "EASCfg": {"url": eas_url, "token": eas_token},
            "HOLOCfg": {
                "PG_HOST": pg_host,
                "PG_DATABASE": pg_database,
                "PG_PORT": 80,
                "PG_USER": pg_user,
                "PG_PASSWORD": pg_pwd,
                "TABLE": table,
            },
            "create_docs": {
                "chunk_size": 200,
                "chunk_overlap": 0,
                "docs_dir": "docs/",
                "glob": "**/*",
            },
        }
        cfg.update(cfg_db)
        _global_args.vectordb_type = "Hologres"
        _global_cfg.update(cfg)
        _global_args.bm25_load_cache = check_db_cache(
            ["vector_store", "HOLOCfg"], _global_cfg
        )
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect Hologres success."

    def connect_es(
        emb_model, emb_dim, emb_openai_key, es_url, es_index, es_user, es_pwd
    ):
        cfg = get_llm_cfg(llm_src, eas_url, eas_token, open_api_key)
        cfg_db = {
            "embedding": {
                "embedding_model": emb_model,
                "model_dir": "./embedding_model/",
                "embedding_dimension": emb_dim,
                "openai_key": emb_openai_key,
            },
            "EASCfg": {"url": eas_url, "token": eas_token},
            "ElasticSearchCfg": {
                "ES_URL": es_url,
                "ES_INDEX": es_index,
                "ES_USER": es_user,
                "ES_PASSWORD": es_pwd,
            },
            "create_docs": {
                "chunk_size": 200,
                "chunk_overlap": 0,
                "docs_dir": "docs/",
                "glob": "**/*",
            },
        }
        cfg.update(cfg_db)
        _global_args.vectordb_type = "ElasticSearch"
        _global_cfg.update(cfg)
        _global_args.bm25_load_cache = check_db_cache(
            ["vector_store", "ElasticSearchCfg"], _global_cfg
        )
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect ElasticSearch success."

    def connect_faiss(emb_model, emb_dim, emb_openai_key, path, name):
        cfg = get_llm_cfg(llm_src, eas_url, eas_token, open_api_key)
        cfg_db = {
            "embedding": {
                "model_dir": "./embedding_model/",
                "embedding_model": emb_model,
                "embedding_dimension": emb_dim,
                "openai_key": emb_openai_key,
            },
            "EASCfg": {"url": eas_url, "token": eas_token},
            "FAISS": {"index_path": path, "index_name": name},
            "create_docs": {
                "chunk_size": 200,
                "chunk_overlap": 0,
                "docs_dir": "docs/",
                "glob": "**/*",
            },
        }
        cfg.update(cfg_db)
        _global_args.vectordb_type = "FAISS"
        _global_cfg.update(cfg)
        _global_args.bm25_load_cache = check_db_cache(
            ["vector_store", "FAISS"], _global_cfg
        )
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect FAISS success."

    def connect_milvus(
        emb_model,
        emb_dim,
        emb_openai_key,
        milvus_collection,
        milvus_host,
        milvus_port,
        milvus_user,
        milvus_pwd,
        milvus_drop,
    ):
        cfg = get_llm_cfg(
            llm_src,
            eas_url,
            eas_token,
            open_api_key,
        )
        cfg_db = {
            "embedding": {
                "embedding_model": emb_model,
                "model_dir": "./embedding_model/",
                "embedding_dimension": emb_dim,
                "openai_key": emb_openai_key,
            },
            "MilvusCfg": {
                "COLLECTION": milvus_collection,
                "HOST": milvus_host,
                "PORT": milvus_port,
                "USER": milvus_user,
                "PASSWORD": milvus_pwd,
                "DROP": milvus_drop,
            },
            "create_docs": {
                "chunk_size": 200,
                "chunk_overlap": 0,
                "docs_dir": "docs/",
                "glob": "**/*",
            },
        }
        cfg.update(cfg_db)
        _global_args.vectordb_type = "Milvus"
        _global_cfg.update(cfg)
        _global_args.bm25_load_cache = check_db_cache(
            ["vector_store", "MilvusCfg"], _global_cfg
        )
        service.init_with_cfg(_global_cfg, _global_args)
        return "Connect Milvus success."

    with gr.Blocks() as demo:

        value_md = """
            #  <center> \N{fire} Chatbot Langchain with LLM on PAI ! 

            ### <center> \N{rocket} Build your own personalized knowledge base question-answering chatbot. 
                        
            <center> 
            
            \N{fire} Platform: [PAI](https://help.aliyun.com/zh/pai)  /  [PAI-EAS](https://www.aliyun.com/product/bigdata/learn/eas)  / [PAI-DSW](https://pai.console.aliyun.com/notebook) &emsp;  \N{rocket} Supported VectorStores:  [Hologres](https://www.aliyun.com/product/bigdata/hologram)  /  [ElasticSearch](https://www.aliyun.com/product/bigdata/elasticsearch)  /  [AnalyticDB](https://www.aliyun.com/product/apsaradb/gpdb)  /  [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss)
                
            """

        gr.Markdown(value=value_md)
        ding_hl = "<div style='text-align: center;'> \N{fire}欢迎加入【PAI】Chatbot-langchain答疑群”群的钉钉群号： 27370042974 </div>"

        gr.HTML(ding_hl, elem_id="ding")

        with gr.Tab("\N{rocket} Settings"):
            with gr.Row():
                with gr.Column():
                    with gr.Column():
                        md_emb = gr.Markdown(
                            value="**Please set your embedding model.**"
                        )
                        emb_model = gr.Dropdown(
                            [
                                "SGPT-125M-weightedmean-nli-bitfit",
                                "text2vec-large-chinese",
                                "text2vec-base-chinese",
                                "paraphrase-multilingual-MiniLM-L12-v2",
                                "OpenAIEmbeddings",
                            ],
                            label="Embedding Model",
                            value=_global_args.embed_model,
                        )
                        emb_dim = gr.Textbox(
                            label="Embedding Dimension", value=_global_args.embed_dim
                        )
                        emb_openai_key = gr.Textbox(
                            visible=False, label="OpenAI API Key", value=""
                        )

                        def change_emb_model(model):
                            if model == "SGPT-125M-weightedmean-nli-bitfit":
                                return {
                                    emb_dim: gr.update(value="768"),
                                    emb_openai_key: gr.update(visible=False),
                                }
                            if model == "text2vec-large-chinese":
                                return {
                                    emb_dim: gr.update(value="1024"),
                                    emb_openai_key: gr.update(visible=False),
                                }
                            if model == "text2vec-base-chinese":
                                return {
                                    emb_dim: gr.update(value="768"),
                                    emb_openai_key: gr.update(visible=False),
                                }
                            if model == "paraphrase-multilingual-MiniLM-L12-v2":
                                return {
                                    emb_dim: gr.update(value="384"),
                                    emb_openai_key: gr.update(visible=False),
                                }
                            if model == "OpenAIEmbeddings":
                                return {
                                    emb_dim: gr.update(value="1536"),
                                    emb_openai_key: gr.update(visible=True),
                                }

                        emb_model.change(
                            fn=change_emb_model,
                            inputs=emb_model,
                            outputs=[emb_dim, emb_openai_key],
                        )

                with gr.Column():
                    with gr.Column():
                        md_vs = gr.Markdown(
                            value=f"**Please check your Vector Store for {_global_cfg['vector_store']}.**"
                        )
                        logger.info(
                            f"_global_cfg vector_store {_global_cfg['vector_store']}"
                        )
                        if _global_cfg["vector_store"] == "AnalyticDB":
                            with gr.Column() as adb_col:
                                pg_host = gr.Textbox(
                                    label="Host",
                                    value=(
                                        _global_cfg["ADBCfg"]["PG_HOST"]
                                        if _global_cfg["vector_store"] == "AnalyticDB"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                pg_user = gr.Textbox(
                                    label="User",
                                    value=(
                                        _global_cfg["ADBCfg"]["PG_USER"]
                                        if _global_cfg["vector_store"] == "AnalyticDB"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                pg_database = gr.Textbox(
                                    label="Database",
                                    value=(
                                        "postgres"
                                        if _global_cfg["vector_store"] == "AnalyticDB"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                pg_pwd = gr.Textbox(
                                    label="Password",
                                    value=(
                                        _global_cfg["ADBCfg"]["PG_PASSWORD"]
                                        if _global_cfg["vector_store"] == "AnalyticDB"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                pg_collection = gr.Textbox(
                                    label="CollectionName",
                                    value=(
                                        _global_cfg["ADBCfg"]["PG_COLLECTION_NAME"]
                                        if _global_cfg["vector_store"] == "AnalyticDB"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                pg_del = gr.Dropdown(
                                    ["True", "False"],
                                    label="Pre Delete",
                                    value=(
                                        _global_cfg["ADBCfg"]["PRE_DELETE"]
                                        if _global_cfg["vector_store"] == "AnalyticDB"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                connect_btn = gr.Button(
                                    "Connect AnalyticDB", variant="primary"
                                )
                                con_state = gr.Textbox(
                                    label="Connection Info: ", interactive=False
                                )
                                connect_btn.click(
                                    fn=connect_adb,
                                    inputs=[
                                        emb_model,
                                        emb_dim,
                                        emb_openai_key,
                                        pg_host,
                                        pg_user,
                                        pg_pwd,
                                        pg_database,
                                        pg_collection,
                                        pg_del,
                                    ],
                                    outputs=con_state,
                                    api_name="connect_adb",
                                )
                        elif _global_cfg["vector_store"] == "Hologres":
                            with gr.Column() as holo_col:
                                holo_host = gr.Textbox(
                                    label="Host",
                                    value=(
                                        _global_cfg["HOLOCfg"]["PG_HOST"]
                                        if _global_cfg["vector_store"] == "Hologres"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                holo_database = gr.Textbox(
                                    label="Database",
                                    value=(
                                        _global_cfg["HOLOCfg"]["PG_DATABASE"]
                                        if _global_cfg["vector_store"] == "Hologres"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                holo_user = gr.Textbox(
                                    label="User",
                                    value=(
                                        _global_cfg["HOLOCfg"]["PG_USER"]
                                        if _global_cfg["vector_store"] == "Hologres"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                holo_pwd = gr.Textbox(
                                    label="Password",
                                    value=(
                                        _global_cfg["HOLOCfg"]["PG_PASSWORD"]
                                        if _global_cfg["vector_store"] == "Hologres"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                holo_table = gr.Textbox(
                                    label="Table",
                                    value=(
                                        _global_cfg["HOLOCfg"]["TABLE"]
                                        if _global_cfg["vector_store"] == "Hologres"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                connect_btn = gr.Button(
                                    "Connect Hologres", variant="primary"
                                )
                                con_state = gr.Textbox(
                                    label="Connection Info: ", interactive=False
                                )
                                connect_btn.click(
                                    fn=connect_holo,
                                    inputs=[
                                        emb_model,
                                        emb_dim,
                                        emb_openai_key,
                                        holo_host,
                                        holo_database,
                                        holo_user,
                                        holo_pwd,
                                        holo_table,
                                    ],
                                    outputs=con_state,
                                    api_name="connect_holo",
                                )
                        elif _global_cfg["vector_store"] == "ElasticSearch":
                            with gr.Column() as es_col:
                                es_url = gr.Textbox(
                                    label="URL",
                                    value=(
                                        _global_cfg["ElasticSearchCfg"]["ES_URL"]
                                        if _global_cfg["vector_store"]
                                        == "ElasticSearch"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                es_index = gr.Textbox(
                                    label="Index",
                                    value=(
                                        _global_cfg["ElasticSearchCfg"]["ES_INDEX"]
                                        if _global_cfg["vector_store"]
                                        == "ElasticSearch"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                es_user = gr.Textbox(
                                    label="User",
                                    value=(
                                        _global_cfg["ElasticSearchCfg"]["ES_USER"]
                                        if _global_cfg["vector_store"]
                                        == "ElasticSearch"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                es_pwd = gr.Textbox(
                                    label="Password",
                                    value=(
                                        _global_cfg["ElasticSearchCfg"]["ES_PASSWORD"]
                                        if _global_cfg["vector_store"]
                                        == "ElasticSearch"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                connect_btn = gr.Button(
                                    "Connect ElasticSearch", variant="primary"
                                )
                                con_state = gr.Textbox(
                                    label="Connection Info: ", interactive=False
                                )
                                connect_btn.click(
                                    fn=connect_es,
                                    inputs=[
                                        emb_model,
                                        emb_dim,
                                        emb_openai_key,
                                        es_url,
                                        es_index,
                                        es_user,
                                        es_pwd,
                                    ],
                                    outputs=con_state,
                                    api_name="connect_es",
                                )
                        elif _global_cfg["vector_store"] == "FAISS":
                            with gr.Column() as faiss_col:
                                faiss_path = gr.Textbox(
                                    label="Path",
                                    value=(
                                        _global_cfg["FAISS"]["index_path"]
                                        if _global_cfg["vector_store"] == "FAISS"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                faiss_name = gr.Textbox(
                                    label="Index",
                                    value=(
                                        _global_cfg["FAISS"]["index_name"]
                                        if _global_cfg["vector_store"] == "FAISS"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                connect_btn_faiss = gr.Button(
                                    "Connect Faiss", variant="primary"
                                )
                                con_state_faiss = gr.Textbox(
                                    label="Connection Info: ", interactive=False
                                )
                                connect_btn_faiss.click(
                                    fn=connect_faiss,
                                    inputs=[
                                        emb_model,
                                        emb_dim,
                                        emb_openai_key,
                                        faiss_path,
                                        faiss_name,
                                    ],
                                    outputs=con_state_faiss,
                                )
                        elif _global_cfg["vector_store"] == "Milvus":
                            with gr.Column() as milvus_col:
                                milvus_collection = gr.Textbox(
                                    label="CollectionName",
                                    value=(
                                        _global_cfg["MilvusCfg"]["COLLECTION"]
                                        if _global_cfg["vector_store"] == "Milvus"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                milvus_host = gr.Textbox(
                                    label="Host",
                                    value=(
                                        _global_cfg["MilvusCfg"]["HOST"]
                                        if _global_cfg["vector_store"] == "Milvus"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                milvus_port = gr.Textbox(
                                    label="Port",
                                    value=(
                                        _global_cfg["MilvusCfg"]["PORT"]
                                        if _global_cfg["vector_store"] == "Milvus"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                milvus_user = gr.Textbox(
                                    label="User",
                                    value=(
                                        _global_cfg["MilvusCfg"]["USER"]
                                        if _global_cfg["vector_store"] == "Milvus"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                milvus_pwd = gr.Textbox(
                                    label="Password",
                                    value=(
                                        _global_cfg["MilvusCfg"]["PASSWORD"]
                                        if _global_cfg["vector_store"] == "Milvus"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                milvus_drop = gr.Dropdown(
                                    ["True", "False"],
                                    label="Drop Old",
                                    value=(
                                        _global_cfg["MilvusCfg"]["DROP"]
                                        if _global_cfg["vector_store"] == "Milvus"
                                        else ""
                                    ),
                                    interactive=False,
                                )
                                connect_btn = gr.Button(
                                    "Connect Milvus", variant="primary"
                                )
                                con_state = gr.Textbox(
                                    label="Connection Info: ", interactive=False
                                )
                                connect_btn.click(
                                    fn=connect_milvus,
                                    inputs=[
                                        emb_model,
                                        emb_dim,
                                        emb_openai_key,
                                        milvus_collection,
                                        milvus_host,
                                        milvus_port,
                                        milvus_user,
                                        milvus_pwd,
                                        milvus_drop,
                                    ],
                                    outputs=con_state,
                                    api_name="connect_milvus",
                                )

        with gr.Tab("\N{whale} Upload"):
            with gr.Row():
                with gr.Column(scale=2):
                    ft_radio = gr.Dropdown(
                        ["text"],
                        label="Which type of files do you want to upload?",
                        value=_global_cfg["file_type"],
                    )
                    with gr.Column(visible=False) as html_col:
                        rank_radio = gr.Dropdown(
                            ["h1", "h2", "h3", "h4", "h5"],
                            label="Rank Label",
                            value="h2",
                        )
                        qa_model = gr.Radio(
                            ["Yes"], label="With QA Extraction Model", value="Yes"
                        )
                    with gr.Column() as docs_col:
                        chunk_size = gr.Textbox(
                            label="\N{rocket} Chunk Size (The size of the chunks into which a document is divided)",
                            value="200",
                        )
                        chunk_overlap = gr.Textbox(
                            label="\N{fire} Chunk Overlap (The portion of adjacent document chunks that overlap with each other)",
                            value="0",
                        )

                def isFileValid(file_name, types):
                    for t in types:
                        if file_name.endswith(t):
                            return True
                    return False

                def upload_knowledge(
                    upload_file, ft_radio, chunk_size, chunk_overlap, rank_radio
                ):
                    file_name = ""
                    valid_types = (
                        [".txt", ".md", ".docx", ".doc", ".pdf"]
                        if ft_radio == "text"
                        else [".html"]
                    )
                    for file in upload_file:
                        if isFileValid(file.name.lower(), valid_types):
                            file_path = file.name
                            file_name += file.name.rsplit("/", 1)[-1] + ", "
                            service.upload_custom_knowledge(
                                file_path,
                                ft_radio,
                                int(chunk_size),
                                int(chunk_overlap),
                                rank_radio,
                            )
                    return (
                        "Upload "
                        + str(len(upload_file))
                        + " files [ "
                        + file_name
                        + "] Success! \n \n Relevant content has been added to the vector store, you can now start chatting and asking questions."
                    )

                def upload_knowledge_dir(
                    upload_dir, ft_radio, chunk_size, chunk_overlap, rank_radio
                ):
                    valid_types = (
                        [".txt", ".md", ".docx", ".doc", ".pdf"]
                        if ft_radio == "text"
                        else [".html"]
                    )
                    for file in upload_dir:
                        if isFileValid(file.name.lower(), valid_types):
                            file_path = file.name
                            service.upload_custom_knowledge(
                                file_path,
                                ft_radio,
                                int(chunk_size),
                                int(chunk_overlap),
                                rank_radio,
                            )
                    return (
                        "Directory: Upload " + str(len(upload_dir)) + " files Success!"
                    )

                with gr.Column(
                    scale=8, visible=(_global_cfg["file_type"] == "html")
                ) as html_upload_col:
                    with gr.Tab("Files"):
                        upload_file = gr.File(
                            label="Upload a knowledge file (supported type: html)",
                            file_types=[".html"],
                            file_count="multiple",
                        )
                        connect_btn = gr.Button("Upload", variant="primary")
                        state_hl_file = gr.Textbox(label="Upload State")
                    with gr.Tab("Directory"):
                        upload_file_dir = gr.File(
                            label="Upload a knowledge directory (supported type: html)",
                            file_count="directory",
                        )
                        connect_dir_btn = gr.Button("Upload", variant="primary")
                        state_hl_dir = gr.Textbox(label="Upload State")
                    connect_btn.click(
                        fn=upload_knowledge,
                        inputs=[
                            upload_file,
                            ft_radio,
                            chunk_size,
                            chunk_overlap,
                            rank_radio,
                        ],
                        outputs=state_hl_file,
                        api_name="upload_knowledge",
                    )
                    connect_dir_btn.click(
                        fn=upload_knowledge_dir,
                        inputs=[
                            upload_file_dir,
                            ft_radio,
                            chunk_size,
                            chunk_overlap,
                            rank_radio,
                        ],
                        outputs=state_hl_dir,
                        api_name="upload_knowledge_dir",
                    )
                with gr.Column(
                    scale=8, visible=(_global_cfg["file_type"] == "text")
                ) as docs_upload_col:
                    with gr.Tab("Files"):
                        upload_file = gr.File(
                            label="Upload a knowledge file (supported type: txt, md, doc, docx, pdf)",
                            file_types=[".txt", ".md", ".docx", ".pdf", "doc"],
                            file_count="multiple",
                        )
                        connect_btn = gr.Button("Upload", variant="primary")
                        state_hl_file = gr.Textbox(label="Upload State")
                    with gr.Tab("Directory"):
                        upload_file_dir = gr.File(
                            label="Upload a knowledge directory (supported type: txt, md, docx, pdf)",
                            file_count="directory",
                        )
                        connect_dir_btn = gr.Button("Upload", variant="primary")
                        state_hl_dir = gr.Textbox(label="Upload State")
                    connect_btn.click(
                        fn=upload_knowledge,
                        inputs=[
                            upload_file,
                            ft_radio,
                            chunk_size,
                            chunk_overlap,
                            rank_radio,
                        ],
                        outputs=state_hl_file,
                        api_name="upload_knowledge_1",
                    )
                    connect_dir_btn.click(
                        fn=upload_knowledge_dir,
                        inputs=[
                            upload_file_dir,
                            ft_radio,
                            chunk_size,
                            chunk_overlap,
                            rank_radio,
                        ],
                        outputs=state_hl_dir,
                        api_name="upload_knowledge_dir_1",
                    )

        with gr.Tab("\N{fire} Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    ds_radio = gr.Radio(
                        ["Vector Store", "LLM", "Langchain(Vector Store + LLM)"],
                        label="\N{fire} Which query do you want to use?",
                    )

                    with gr.Column(visible=False) as vs_col:
                        vec_model_argument = gr.Accordion(
                            "Parameters of Vector Retrieval"
                        )

                        def change_score_threshold(emb_model):
                            if emb_model == "OpenAIEmbeddings":
                                return {
                                    score_threshold: gr.update(
                                        maximum=1,
                                        step=0.01,
                                        value=0.5,
                                        label="Score Threshold (choose between 0 and 1, the more similar the vectors, the smaller the value.)",
                                    )
                                }
                            else:
                                return {
                                    score_threshold: gr.update(
                                        maximum=1000,
                                        step=0.1,
                                        value=5,
                                        label="Similarity Distance Threshold (The more similar the vectors, the smaller the value.)",
                                    )
                                }

                        with vec_model_argument:
                            topk = gr.Slider(
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=3,
                                label="Top K (choose between 0 and 100)",
                            )
                            score_threshold = gr.Slider(
                                minimum=0,
                                maximum=1000,
                                step=0.1,
                                value=200,
                                label="Similarity Distance Threshold (The more similar the vectors, the smaller the value.)",
                            )
                            rerank_model = gr.Radio(
                                [
                                    "No Re-Rank",
                                    "BGE-Reranker-Base",
                                    "BGE-Reranker-Large",
                                ],
                                label="Re-Rank Model (Note: It will take a long time to load the model when using it for the first time.)",
                                value="No Re-Rank",
                            )
                            kw_retrieval = gr.Radio(
                                ["Embedding Only", "Keyword Ensembled"],
                                label="Keyword Retrieval",
                                value="Embedding Only",
                            )
                            emb_model.change(
                                fn=change_score_threshold,
                                inputs=emb_model,
                                outputs=[score_threshold],
                            )

                    with gr.Column(visible=False) as llm_col:
                        model_argument = gr.Accordion("Inference Parameters of LLM")
                        with model_argument:
                            llm_topk = gr.Slider(
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=30,
                                label="Top K (choose between 0 and 100)",
                            )
                            llm_topp = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.8,
                                label="Top P (choose between 0 and 1)",
                            )
                            llm_temp = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.7,
                                label="Temperature (choose between 0 and 1)",
                            )
                            history_radio = gr.Radio(
                                ["Yes", "No"], value="No", label="With Chat History"
                            )

                    with gr.Column(visible=False) as lc_col:
                        prm_radio = gr.Radio(
                            [
                                "Simple",
                                "General",
                                "Extract URL",
                                "Accurate Content",
                                "Customize",
                            ],
                            label="\N{rocket} Please choose the prompt template type",
                            value="Simple",
                        )
                        prompt = gr.Textbox(
                            label="prompt template",
                            placeholder="This is a prompt template",
                            lines=4,
                        )

                        def change_prompt_template(prm_radio):
                            if prm_radio == "Simple":
                                return {
                                    prompt: gr.update(
                                        value="参考内容如下：\n{context}\n作为个人知识答疑助手，请根据上述参考内容回答下面问题，答案中不允许包含编造内容。\n用户问题:\n{question}"
                                    )
                                }
                            elif prm_radio == "General":
                                return {
                                    prompt: gr.update(
                                        value='基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。\n=====\n已知信息:\n{context}\n=====\n用户问题:\n{question}'
                                    )
                                }
                            elif prm_radio == "Extract URL":
                                return {
                                    prompt: gr.update(
                                        value="你是一位智能小助手，请根据下面我所提供的相关知识，对我提出的问题进行回答。回答的内容必须包括其定义、特征、应用领域以及相关网页链接等等内容，同时务必满足下方所提的要求！\n=====\n 知识库相关知识如下:\n{context}\n=====\n 请根据上方所提供的知识库内容与要求，回答以下问题:\n {question}"
                                    )
                                }
                            elif prm_radio == "Accurate Content":
                                return {
                                    prompt: gr.update(
                                        value="你是一位知识小助手，请根据下面我提供的知识库中相关知识，对我提出的若干问题进行回答，同时回答的内容需满足我所提的要求! \n=====\n 知识库相关知识如下:\n{context}\n=====\n 请根据上方所提供的知识库内容与要求，回答以下问题:\n {question}"
                                    )
                                }
                            elif prm_radio == "Customize":
                                return {prompt: gr.update(value="")}

                        prm_radio.change(
                            fn=change_prompt_template,
                            inputs=prm_radio,
                            outputs=[prompt],
                        )
                    cur_tokens = gr.Textbox(
                        label="\N{fire} Current total count of tokens"
                    )

                    def change_query_radio(ds_radio):
                        if ds_radio == "Vector Store":
                            return {
                                vs_col: gr.update(visible=True),
                                llm_col: gr.update(visible=False),
                                lc_col: gr.update(visible=False),
                            }
                        elif ds_radio == "LLM":
                            return {
                                vs_col: gr.update(visible=False),
                                llm_col: gr.update(visible=True),
                                lc_col: gr.update(visible=False),
                            }
                        elif ds_radio == "Langchain(Vector Store + LLM)":
                            return {
                                vs_col: gr.update(visible=True),
                                llm_col: gr.update(visible=True),
                                lc_col: gr.update(visible=True),
                            }

                    ds_radio.change(
                        fn=change_query_radio,
                        inputs=ds_radio,
                        outputs=[vs_col, llm_col, lc_col],
                    )

                with gr.Column(scale=8):
                    chatbot = gr.Chatbot(height=500)
                    msg = gr.Textbox(label="Enter your question.")
                    with gr.Row():
                        submitBtn = gr.Button("Submit", variant="primary")
                        summaryBtn = gr.Button("Summary", variant="primary")
                        clear_his = gr.Button("Clear History", variant="secondary")
                        clear = gr.ClearButton([msg, chatbot])

                    def respond(
                        message,
                        chat_history,
                        ds_radio,
                        topk,
                        score_threshold,
                        rerank_model,
                        kw_retrieval,
                        llm_topk,
                        llm_topp,
                        llm_temp,
                        prm_radio,
                        prompt,
                        history_radio,
                    ):
                        summary_res = ""
                        history = False
                        if history_radio == "Yes":
                            history = True
                        if ds_radio == "Vector Store":
                            if message.strip() == "":
                                answer = "Please enter your question, the question cannot be empty."
                                lens = 0
                            else:
                                answer, lens = service.query_only_vectorstore(
                                    message,
                                    topk,
                                    score_threshold,
                                    rerank_model,
                                    kw_retrieval,
                                )
                        elif ds_radio == "LLM":
                            answer, lens, summary_res = service.query_only_llm(
                                message, history, llm_topk, llm_topp, llm_temp
                            )
                        else:
                            if message.strip() == "":
                                answer = "Please enter your question, the question cannot be empty."
                                lens = 0
                            else:
                                answer, lens, summary_res = service.query_retrieval_llm(
                                    message,
                                    topk,
                                    score_threshold,
                                    rerank_model,
                                    kw_retrieval,
                                    prm_radio,
                                    prompt,
                                    history,
                                    llm_topk,
                                    llm_topp,
                                    llm_temp,
                                )
                        bot_message = answer
                        chat_history.append((message, bot_message))
                        time.sleep(0.05)
                        return "", chat_history, str(lens) + "\n" + summary_res

                    def clear_hisoty(chat_history):
                        chat_history = []
                        service.langchain_chat_history = []
                        service.input_tokens = []
                        time.sleep(0.05)
                        return chat_history, "0 \n Clear history successfully!"

                    def summary_hisoty(chat_history):
                        service.input_tokens = []
                        bot_message = service.checkout_history_and_summary(summary=True)
                        chat_history.append(
                            ("请对我们之前的对话内容进行总结。", bot_message)
                        )
                        tokens_len = service.sp.encode(
                            service.input_tokens, out_type=str
                        )
                        lens = sum(len(tl) for tl in tokens_len)
                        time.sleep(0.05)
                        return chat_history, str(lens) + "\n" + bot_message

                    submitBtn.click(
                        respond,
                        [
                            msg,
                            chatbot,
                            ds_radio,
                            topk,
                            score_threshold,
                            rerank_model,
                            kw_retrieval,
                            llm_topk,
                            llm_topp,
                            llm_temp,
                            prm_radio,
                            prompt,
                            history_radio,
                        ],
                        [msg, chatbot, cur_tokens],
                    )
                    clear_his.click(clear_hisoty, [chatbot], [chatbot, cur_tokens])
                    summaryBtn.click(summary_hisoty, [chatbot], [chatbot, cur_tokens])

        footer = html("footer.html")
        gr.HTML(footer, elem_id="footer")

    return demo
