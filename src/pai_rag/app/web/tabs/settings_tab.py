from typing import Dict, Any, List
import gradio as gr
import datetime
import traceback
from pai_rag.app.web.ui_constants import (
    EMBEDDING_API_KEY_DICT,
    DEFAULT_EMBED_SIZE,
    EMBEDDING_DIM_DICT,
    LLM_MODEL_KEY_DICT,
)
from pai_rag.app.web.rag_client import rag_client
from pai_rag.app.web.view_model import view_model
from pai_rag.app.web.utils import components_to_dict
from pai_rag.app.web.tabs.vector_db_panel import create_vector_db_panel
import logging

logger = logging.getLogger(__name__)


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


def create_setting_tab() -> Dict[str, Any]:
    components = []
    with gr.Row():
        with gr.Column():
            with gr.Column():
                _ = gr.Markdown(value="**Please choose your embedding model.**")
                embed_source = gr.Dropdown(
                    EMBEDDING_API_KEY_DICT.keys(),
                    label="Embedding Type",
                    elem_id="embed_source",
                )
                embed_model = gr.Dropdown(
                    EMBEDDING_DIM_DICT.keys(),
                    label="Embedding Model Name",
                    elem_id="embed_model",
                    visible=False,
                )
                embed_dim = gr.Textbox(
                    label="Embedding Dimension",
                    elem_id="embed_dim",
                )

                def change_emb_source(source):
                    view_model.embed_source = source
                    return {
                        embed_model: gr.update(visible=(source == "HuggingFace")),
                        embed_dim: EMBEDDING_DIM_DICT.get(
                            view_model.embed_model, DEFAULT_EMBED_SIZE
                        ),
                    }

                def change_emb_model(model):
                    view_model.embed_model = model
                    return {
                        embed_dim: EMBEDDING_DIM_DICT.get(
                            view_model.embed_model, DEFAULT_EMBED_SIZE
                        )
                    }

                embed_source.change(
                    fn=change_emb_source,
                    inputs=embed_source,
                    outputs=[embed_model, embed_dim],
                )
                embed_model.change(
                    fn=change_emb_model,
                    inputs=embed_model,
                    outputs=[embed_dim],
                )
            components.extend([embed_source, embed_dim, embed_model])

            with gr.Column():
                _ = gr.Markdown(value="**Please set your LLM.**")
                llm = gr.Dropdown(
                    ["PaiEas", "OpenAI", "DashScope"],
                    label="LLM Model Source",
                    elem_id="llm",
                )
                with gr.Column(visible=(view_model.llm == "PaiEas")) as eas_col:
                    llm_eas_url = gr.Textbox(
                        label="EAS Url",
                        elem_id="llm_eas_url",
                    )
                    llm_eas_token = gr.Textbox(
                        label="EAS Token",
                        elem_id="llm_eas_token",
                    )
                    llm_eas_model_name = gr.Textbox(
                        label="EAS Model name",
                        elem_id="llm_eas_model_name",
                    )
                with gr.Column(visible=(view_model.llm != "PaiEas")) as api_llm_col:
                    llm_api_model_name = gr.Dropdown(
                        label="LLM Model Name",
                        elem_id="llm_api_model_name",
                    )

                components.extend(
                    [
                        llm,
                        llm_eas_url,
                        llm_eas_token,
                        llm_eas_model_name,
                        llm_api_model_name,
                    ]
                )

                def change_llm(value):
                    view_model.llm = value
                    eas_visible = value == "PaiEas"
                    api_visible = value != "PaiEas"
                    model_options = LLM_MODEL_KEY_DICT.get(value, [])
                    cur_model = model_options[0] if model_options else ""
                    return {
                        eas_col: gr.update(visible=eas_visible),
                        api_llm_col: gr.update(visible=api_visible),
                        llm_api_model_name: gr.update(
                            choices=model_options, value=cur_model
                        ),
                    }

                llm.change(
                    fn=change_llm,
                    inputs=llm,
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
        vector_db_elems = create_vector_db_panel(
            input_elements={
                llm,
                llm_eas_url,
                llm_eas_token,
                llm_eas_model_name,
                embed_source,
                embed_model,
                embed_dim,
                llm_api_model_name,
            },
            connect_vector_func=connect_vector_db,
        )

        elems = components_to_dict(components)
        elems.update(vector_db_elems)
        return elems
