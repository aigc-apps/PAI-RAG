import datetime
import gradio as gr
import os
from typing import List, Any
from pai_rag.app.web.view_model import view_model
from pai_rag.app.web.rag_client import rag_client
from pai_rag.app.web.vector_db_panel import create_vector_db_panel
from pai_rag.app.web.prompts import (
    SIMPLE_PROMPTS,
    GENERAL_PROMPTS,
    EXTRACT_URL_PROMPTS,
    ACCURATE_CONTENT_PROMPTS,
    PROMPT_MAP,
)
from os import environ

import logging
import traceback

logger = logging.getLogger("WebUILogger")

welcome_message_markdown = """
            # \N{fire} Chatbot with RAG on PAI !
            ### \N{rocket} Build your own personalized knowledge base question-answering chatbot.

            #### \N{fire} Platform: [PAI](https://help.aliyun.com/zh/pai)  /  [PAI-EAS](https://www.aliyun.com/product/bigdata/learn/eas)  / [PAI-DSW](https://pai.console.aliyun.com/notebook) &emsp;  \N{rocket} Supported VectorStores:  [Hologres](https://www.aliyun.com/product/bigdata/hologram)  /  [ElasticSearch](https://www.aliyun.com/product/bigdata/elasticsearch)  /  [AnalyticDB](https://www.aliyun.com/product/apsaradb/gpdb)  /  [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss)

            #### \N{fire} <a href='/docs'>API Docs</a> &emsp; \N{rocket} \N{fire}  欢迎加入【PAI】RAG答疑群 27370042974
            """

css_style = """
        h1, h3, h4 {
            text-align: center;
            display:block;
        }
        """

DEFAULT_EMBED_SIZE = 1536

embedding_dim_dict = {
    "bge-small-zh-v1.5": 1024,
    "SGPT-125M-weightedmean-nli-bitfit": 768,
    "text2vec-large-chinese": 1024,
    "text2vec-base-chinese": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
}

embedding_api_key_dict = {"HuggingFace": False, "OpenAI": True, "DashScope": True}

llm_model_key_dict = {
    "DashScope": [
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
        "qwen-max-1201",
        "qwen-max-longcontext",
    ],
    "OpenAI": [
        "gpt-3.5-turbo",
        "gpt-4-turbo",
    ],
}

current_session_id = None


def connect_vector_db(input_elements: List[Any]):
    try:
        update_dict = {}
        for element, value in input_elements.items():
            update_dict[element.elem_id] = value

        view_model.update(update_dict)
        new_config = view_model.to_app_config()
        rag_client.reload_config(new_config)
        return f"[{datetime.datetime.now()}] Connect vector db success!"
    except Exception as ex:
        logger.critical(f"[Critical] Connect failed. {traceback.format_exc()}")
        return f"Connect failed. Please check: {ex}"


def upload_knowledge(upload_files, chunk_size, chunk_overlap, enable_qa_extraction):
    view_model.chunk_size = chunk_size
    view_model.chunk_overlap = chunk_overlap
    new_config = view_model.to_app_config()
    rag_client.reload_config(new_config)

    if not upload_files:
        return "No file selected. Please choose at least one file."

    for file in upload_files:
        file_dir = os.path.dirname(file.name)
        rag_client.add_knowledge(file_dir, enable_qa_extraction)
    return (
        "Upload "
        + str(len(upload_files))
        + " files Success! \n \n Relevant content has been added to the vector store, you can now start chatting and asking questions."
    )


def clear_history(chatbot):
    chatbot = []
    global current_session_id
    current_session_id = None
    return chatbot, 0


def respond(input_elements: List[Any]):
    global current_session_id

    update_dict = {}
    for element, value in input_elements.items():
        update_dict[element.elem_id] = value

    # empty input.
    if not update_dict["question"]:
        return "", update_dict["chatbot"], 0

    view_model.update(update_dict)
    new_config = view_model.to_app_config()
    rag_client.reload_config(new_config)

    query_type = update_dict["query_type"]
    msg = update_dict["question"]
    chatbot = update_dict["chatbot"]

    if query_type == "LLM":
        response = rag_client.query_llm(
            msg,
            session_id=current_session_id,
        )

    elif query_type == "Retrieval":
        response = rag_client.query_vector(msg)
    else:
        response = rag_client.query(msg, session_id=current_session_id)
    print("history======:", update_dict["include_history"])
    if update_dict["include_history"]:
        current_session_id = response.session_id
    else:
        current_session_id = None
    chatbot.append((msg, response.answer))
    return "", chatbot, 0


def create_ui():
    with gr.Blocks(css=css_style) as homepage:
        gr.Markdown(value=welcome_message_markdown)

        with gr.Tab("\N{rocket} Settings"):
            with gr.Row():
                with gr.Column():
                    with gr.Column():
                        _ = gr.Markdown(value="**Please choose your embedding model.**")
                        embed_source = gr.Dropdown(
                            embedding_api_key_dict.keys(),
                            label="Embedding Type",
                            value=view_model.embed_source,
                            elem_id="embed_source",
                        )
                        embed_model = gr.Dropdown(
                            embedding_dim_dict.keys(),
                            label="Embedding Model Name",
                            value=view_model.embed_model,
                            elem_id="embed_model",
                            visible=(view_model.embed_source == "HuggingFace"),
                        )
                        embed_api_key = gr.Textbox(
                            visible=view_model.embed_source != "HuggingFace",
                            label="Embedding API Key",
                            value=view_model.embed_api_key,
                            type="password",
                            interactive=True,
                            elem_id="embed_api_key",
                        )
                        embed_dim = gr.Textbox(
                            label="Embedding Dimension",
                            value=embedding_dim_dict.get(
                                view_model.embed_model, DEFAULT_EMBED_SIZE
                            ),
                            elem_id="embed_dim",
                        )

                        def change_emb_source(source):
                            view_model.embed_source = source
                            return {
                                embed_model: gr.update(
                                    visible=(source == "HuggingFace")
                                ),
                                embed_dim: embedding_dim_dict.get(
                                    view_model.embed_model, DEFAULT_EMBED_SIZE
                                ),
                                embed_api_key: gr.update(
                                    visible=(source != "HuggingFace")
                                ),
                            }

                        def change_emb_model(model):
                            view_model.embed_model = model
                            return {
                                embed_dim: embedding_dim_dict.get(
                                    view_model.embed_model, DEFAULT_EMBED_SIZE
                                ),
                                embed_api_key: gr.update(
                                    visible=(view_model.embed_source != "HuggingFace")
                                ),
                            }

                        embed_source.change(
                            fn=change_emb_source,
                            inputs=embed_source,
                            outputs=[embed_model, embed_dim, embed_api_key],
                        )
                        embed_model.change(
                            fn=change_emb_model,
                            inputs=embed_model,
                            outputs=[embed_dim, embed_api_key],
                        )

                    with gr.Column():
                        _ = gr.Markdown(value="**Please set your LLM.**")
                        llm_src = gr.Dropdown(
                            ["PaiEas", "OpenAI", "DashScope"],
                            label="LLM Model Source",
                            value=view_model.llm,
                            elem_id="llm",
                        )
                        with gr.Column(visible=(view_model.llm == "PaiEas")) as eas_col:
                            llm_eas_url = gr.Textbox(
                                label="EAS Url",
                                value=view_model.llm_eas_url,
                                elem_id="llm_eas_url",
                            )
                            llm_eas_token = gr.Textbox(
                                label="EAS Token",
                                value=view_model.llm_eas_token,
                                elem_id="llm_eas_token",
                            )
                            llm_eas_model_name = gr.Textbox(
                                label="EAS Model name",
                                value=view_model.llm_eas_model_name,
                                elem_id="llm_eas_model_name",
                            )
                        with gr.Column(
                            visible=(view_model.llm != "PaiEas")
                        ) as api_llm_col:
                            llm_api_key = gr.Textbox(
                                label="API Key",
                                value=view_model.llm_api_key,
                                elem_id="llm_api_key",
                            )
                            llm_api_model_name = gr.Dropdown(
                                llm_model_key_dict.get(view_model.llm, []),
                                label="LLM Model Name",
                                value=view_model.llm_api_model_name,
                                elem_id="llm_api_model_name",
                            )

                        def change_llm_src(value):
                            view_model.llm = value
                            eas_visible = value == "PaiEas"
                            api_visible = value != "PaiEas"
                            model_options = llm_model_key_dict.get(value, [])
                            cur_model = model_options[0] if model_options else ""
                            return {
                                eas_col: gr.update(visible=eas_visible),
                                api_llm_col: gr.update(visible=api_visible),
                                llm_api_model_name: gr.update(
                                    choices=model_options, value=cur_model
                                ),
                            }

                        llm_src.change(
                            fn=change_llm_src,
                            inputs=llm_src,
                            outputs=[eas_col, api_llm_col, llm_api_model_name],
                        )
                    """
                    with gr.Column():
                        _ = gr.Markdown(
                            value="**(Optional) Please upload your config file.**"
                        )
                        config_file = gr.File(
                            value=view_model.config_file,
                            label="Upload a local config json file",
                            file_types=[".json"],
                            file_count="single",
                            interactive=True,
                        )
                        cfg_btn = gr.Button("Parse Config", variant="primary")
                    """
                create_vector_db_panel(
                    view_model=view_model,
                    input_elements={
                        llm_src,
                        llm_eas_url,
                        llm_eas_token,
                        llm_eas_model_name,
                        embed_source,
                        embed_model,
                        embed_dim,
                        llm_api_key,
                        llm_api_model_name,
                    },
                    connect_vector_func=connect_vector_db,
                )

        with gr.Tab("\N{whale} Upload"):
            with gr.Row():
                with gr.Column(scale=2):
                    chunk_size = gr.Textbox(
                        label="\N{rocket} Chunk Size (The size of the chunks into which a document is divided)",
                        value=view_model.chunk_size,
                    )
                    chunk_overlap = gr.Textbox(
                        label="\N{fire} Chunk Overlap (The portion of adjacent document chunks that overlap with each other)",
                        value=view_model.chunk_overlap,
                    )
                    enable_qa_extraction = gr.Checkbox(
                        label="Yes",
                        info="Process with QA Extraction Model",
                        value=view_model.enable_qa_extraction,
                        elem_id="enable_qa_extraction",
                    )
                with gr.Column(scale=8):
                    with gr.Tab("Files"):
                        upload_file = gr.File(
                            label="Upload a knowledge file.", file_count="multiple"
                        )
                        upload_file_btn = gr.Button("Upload", variant="primary")
                        upload_file_state = gr.Textbox(label="Upload State")
                    with gr.Tab("Directory"):
                        upload_file_dir = gr.File(
                            label="Upload a knowledge directory.",
                            file_count="directory",
                        )
                        upload_dir_btn = gr.Button("Upload", variant="primary")
                        upload_dir_state = gr.Textbox(label="Upload State")
                    upload_file_btn.click(
                        fn=upload_knowledge,
                        inputs=[
                            upload_file,
                            chunk_size,
                            chunk_overlap,
                            enable_qa_extraction,
                        ],
                        outputs=upload_file_state,
                        api_name="upload_knowledge",
                    )
                    upload_dir_btn.click(
                        fn=upload_knowledge,
                        inputs=[
                            upload_file_dir,
                            chunk_size,
                            chunk_overlap,
                            enable_qa_extraction,
                        ],
                        outputs=upload_dir_state,
                        api_name="upload_knowledge_dir",
                    )

        with gr.Tab("\N{fire} Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_type = gr.Radio(
                        ["Retrieval", "LLM", "RAG (Retrieval + LLM)"],
                        label="\N{fire} Which query do you want to use?",
                        elem_id="query_type",
                        value="RAG (Retrieval + LLM)",
                    )

                    with gr.Column(visible=True) as vs_col:
                        vec_model_argument = gr.Accordion(
                            "Parameters of Vector Retrieval"
                        )

                        with vec_model_argument:
                            similarity_top_k = gr.Slider(
                                minimum=0,
                                maximum=100,
                                step=1,
                                elem_id="similarity_top_k",
                                value=view_model.similarity_top_k,
                                label="Top K (choose between 0 and 100)",
                            )
                            # similarity_cutoff = gr.Slider(minimum=0, maximum=1, step=0.01,elem_id="similarity_cutoff",value=view_model.similarity_cutoff, label="Similarity Distance Threshold (The more similar the vectors, the smaller the value.)")
                            rerank_model = gr.Radio(
                                ["No Rerank", "bge-reranker-base", "LLMRerank"],
                                label="Re-Rank Model (Note: It will take a long time to load the model when using it for the first time.)",
                                elem_id="rerank_model",
                                value=view_model.rerank_model,
                            )
                            retrieval_mode = gr.Radio(
                                ["Embedding Only", "Keyword Ensembled", "Keyword Only"],
                                label="Retrieval Mode",
                                elem_id="retrieval_mode",
                                value=view_model.retrieval_mode,
                            )
                        vec_args = {
                            similarity_top_k,
                            # similarity_cutoff,
                            rerank_model,
                            retrieval_mode,
                        }
                    with gr.Column(visible=True) as llm_col:
                        model_argument = gr.Accordion("Inference Parameters of LLM")
                        with model_argument:
                            include_history = gr.Checkbox(
                                label="Chat history",
                                info="Query with chat history.",
                                elem_id="include_history",
                            )
                            llm_topk = gr.Slider(
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=30,
                                elem_id="llm_topk",
                                label="Top K (choose between 0 and 100)",
                            )
                            llm_topp = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.8,
                                elem_id="llm_topp",
                                label="Top P (choose between 0 and 1)",
                            )
                            llm_temp = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.7,
                                elem_id="llm_temp",
                                label="Temperature (choose between 0 and 1)",
                            )
                        llm_args = {llm_topk, llm_topp, llm_temp, include_history}

                    with gr.Column(visible=True) as lc_col:
                        prm_type = PROMPT_MAP.get(view_model.text_qa_template, "Custom")
                        prm_radio = gr.Radio(
                            [
                                "Simple",
                                "General",
                                "Extract URL",
                                "Accurate Content",
                                "Custom",
                            ],
                            label="\N{rocket} Please choose the prompt template type",
                            value=prm_type,
                        )
                        text_qa_template = gr.Textbox(
                            label="prompt template",
                            placeholder=view_model.text_qa_template,
                            value=view_model.text_qa_template,
                            elem_id="text_qa_template",
                            lines=4,
                        )

                        def change_prompt_template(prm_radio):
                            if prm_radio == "Simple":
                                return {
                                    text_qa_template: gr.update(
                                        value=SIMPLE_PROMPTS, interactive=False
                                    )
                                }
                            elif prm_radio == "General":
                                return {
                                    text_qa_template: gr.update(
                                        value=GENERAL_PROMPTS, interactive=False
                                    )
                                }
                            elif prm_radio == "Extract URL":
                                return {
                                    text_qa_template: gr.update(
                                        value=EXTRACT_URL_PROMPTS, interactive=False
                                    )
                                }
                            elif prm_radio == "Accurate Content":
                                return {
                                    text_qa_template: gr.update(
                                        value=ACCURATE_CONTENT_PROMPTS,
                                        interactive=False,
                                    )
                                }
                            else:
                                return {
                                    text_qa_template: gr.update(
                                        value="", interactive=True
                                    )
                                }

                        prm_radio.change(
                            fn=change_prompt_template,
                            inputs=prm_radio,
                            outputs=[text_qa_template],
                        )

                    cur_tokens = gr.Textbox(
                        label="\N{fire} Current total count of tokens", visible=False
                    )

                    def change_query_radio(query_type):
                        global current_session_id
                        current_session_id = None
                        if query_type == "Retrieval":
                            return {
                                vs_col: gr.update(visible=True),
                                llm_col: gr.update(visible=False),
                                lc_col: gr.update(visible=False),
                            }
                        elif query_type == "LLM":
                            return {
                                vs_col: gr.update(visible=False),
                                llm_col: gr.update(visible=True),
                                lc_col: gr.update(visible=False),
                            }
                        elif query_type == "RAG (Retrieval + LLM)":
                            return {
                                vs_col: gr.update(visible=True),
                                llm_col: gr.update(visible=True),
                                lc_col: gr.update(visible=True),
                            }

                    query_type.change(
                        fn=change_query_radio,
                        inputs=query_type,
                        outputs=[vs_col, llm_col, lc_col],
                    )

                with gr.Column(scale=8):
                    chatbot = gr.Chatbot(height=500, elem_id="chatbot")
                    question = gr.Textbox(
                        label="Enter your question.", elem_id="question"
                    )
                    with gr.Row():
                        submitBtn = gr.Button("Submit", variant="primary")
                        clearBtn = gr.Button("Clear History", variant="secondary")

                chat_args = (
                    {text_qa_template, question, query_type, chatbot}
                    .union(vec_args)
                    .union(llm_args)
                )

                submitBtn.click(
                    respond,
                    chat_args,
                    [question, chatbot, cur_tokens],
                    api_name="respond",
                )
                clearBtn.click(clear_history, [chatbot], [chatbot, cur_tokens])

    return homepage
