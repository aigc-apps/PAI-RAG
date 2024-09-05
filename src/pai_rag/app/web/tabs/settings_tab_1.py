from typing import Dict, Any, List
import gradio as gr
import datetime
from pai_rag.app.web.ui_constants import (
    EMBEDDING_API_KEY_DICT,
    DEFAULT_EMBED_SIZE,
    EMBEDDING_DIM_DICT,
    LLM_MODEL_KEY_DICT,
    MLLM_MODEL_KEY_DICT,
)
from pai_rag.app.web.rag_client import RagApiError, rag_client
from pai_rag.app.web.utils import components_to_dict
from pai_rag.app.web.tabs.vector_db_panel import create_vector_db_panel
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_IS_INTERACTIVE = os.environ.get("PAIRAG_RAG__SETTING__interactive", "true")


def connect_vector_db(input_elements: List[Any]):
    try:
        update_dict = {}
        for element, value in input_elements.items():
            update_dict[element.elem_id] = value

        rag_client.patch_config(update_dict)
        return f"[{datetime.datetime.now()}] Connect vector db success!"
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def create_setting_tab() -> Dict[str, Any]:
    components = []
    with gr.Row():
        with gr.Column():
            with gr.Column(scale=5):
                _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **Embedding Model**")
                embed_source = gr.Radio(
                    EMBEDDING_API_KEY_DICT.keys(),
                    label="Embedding Type",
                    elem_id="embed_source",
                    interactive=DEFAULT_IS_INTERACTIVE.lower() != "false",
                )
                embed_model = gr.Dropdown(
                    EMBEDDING_DIM_DICT.keys(),
                    label="Embedding Model Name",
                    elem_id="embed_model",
                    visible=False,
                )
                with gr.Row():
                    embed_dim = gr.Textbox(
                        label="Embedding Dimension",
                        elem_id="embed_dim",
                    )
                    embed_batch_size = gr.Textbox(
                        label="Embedding Batch Size",
                        elem_id="embed_batch_size",
                    )
            with gr.Column(scale=5, variant="panel"):
                _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **(Optional) OSS Bucket**")
                use_oss = gr.Checkbox(
                    label="Use OSS Storage",
                    elem_id="use_oss",
                    container=False,
                )

                with gr.Row(visible=False) as oss_col:
                    oss_ak = gr.Textbox(
                        label="Access Key",
                        elem_id="oss_ak",
                        type="password",
                    )
                    oss_sk = gr.Textbox(
                        label="Access Secret",
                        elem_id="oss_sk",
                        type="password",
                    )
                    oss_bucket = gr.Textbox(
                        label="OSS Bucket",
                        elem_id="oss_bucket",
                    )
                    oss_prefix = gr.Textbox(
                        label="OSS Prefix",
                        elem_id="oss_prefix",
                    )

            def change_emb_source(source):
                return {
                    embed_model: gr.update(visible=(source == "HuggingFace")),
                    embed_dim: EMBEDDING_DIM_DICT.get(source, DEFAULT_EMBED_SIZE)
                    if source == "HuggingFace"
                    else DEFAULT_EMBED_SIZE,
                }

            def change_emb_model(source, model):
                return {
                    embed_dim: EMBEDDING_DIM_DICT.get(model, DEFAULT_EMBED_SIZE)
                    if source == "HuggingFace"
                    else DEFAULT_EMBED_SIZE,
                }

            def change_use_oss(use_oss):
                if use_oss:
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)

            embed_source.input(
                fn=change_emb_source,
                inputs=embed_source,
                outputs=[embed_model, embed_dim],
            )
            embed_model.input(
                fn=change_emb_model,
                inputs=[embed_source, embed_model],
                outputs=[embed_dim],
            )
            use_oss.change(
                fn=change_use_oss,
                inputs=use_oss,
                outputs=oss_col,
            )
            components.extend(
                [
                    embed_source,
                    embed_dim,
                    embed_model,
                    embed_batch_size,
                    oss_ak,
                    oss_sk,
                    oss_bucket,
                    oss_prefix,
                ]
            )

        with gr.Column():
            with gr.Column(scale=5):
                _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **Large Language Model**")
                llm = gr.Radio(
                    ["PaiEas", "DashScope", "OpenAI"],
                    label="LLM Model Source",
                    elem_id="llm",
                    interactive=DEFAULT_IS_INTERACTIVE.lower() != "false",
                )
                with gr.Column(visible=(llm == "PaiEas")) as eas_col:
                    with gr.Row():
                        llm_eas_url = gr.Textbox(
                            label="EAS Url",
                            elem_id="llm_eas_url",
                            interactive=True,
                        )
                        llm_eas_token = gr.Textbox(
                            label="EAS Token",
                            elem_id="llm_eas_token",
                            type="password",
                            interactive=True,
                        )
                        llm_eas_model_name = gr.Textbox(
                            label="EAS Model name",
                            placeholder="Not Required",
                            elem_id="llm_eas_model_name",
                            interactive=True,
                        )
                with gr.Column(visible=(llm != "PaiEas")) as api_llm_col:
                    llm_api_model_name = gr.Dropdown(
                        label="LLM Model Name",
                        elem_id="llm_api_model_name",
                    )
            with gr.Column(scale=5, variant="panel"):
                _ = gr.Markdown(
                    value="\N{WHITE MEDIUM STAR} **(Optional) Multi-Modal Large Language Model**"
                )
                use_mllm = gr.Checkbox(
                    label="Use Multi-Modal LLM",
                    elem_id="use_mllm",
                    container=False,
                )
                with gr.Column(visible=False) as use_mllm_col:
                    mllm = gr.Radio(
                        ["PaiEas", "DashScope"],
                        label="LLM Model Source",
                        elem_id="mllm",
                        interactive=DEFAULT_IS_INTERACTIVE.lower() != "false",
                    )
                    with gr.Column(visible=(mllm == "PaiEas")) as m_eas_col:
                        with gr.Row():
                            mllm_eas_url = gr.Textbox(
                                label="EAS Url",
                                elem_id="mllm_eas_url",
                                interactive=True,
                            )
                            mllm_eas_token = gr.Textbox(
                                label="EAS Token",
                                elem_id="mllm_eas_token",
                                type="password",
                                interactive=True,
                            )
                            mllm_eas_model_name = gr.Textbox(
                                label="EAS Model Name",
                                placeholder="Not Required",
                                elem_id="mllm_eas_model_name",
                                interactive=True,
                            )
                    with gr.Column(visible=(mllm != "PaiEas")) as api_mllm_col:
                        mllm_api_model_name = gr.Dropdown(
                            label="LLM Model Name",
                            elem_id="mllm_api_model_name",
                        )

                components.extend(
                    [
                        llm,
                        llm_eas_url,
                        llm_eas_token,
                        llm_eas_model_name,
                        llm_api_model_name,
                        use_mllm,
                        mllm,
                        mllm_eas_url,
                        mllm_eas_token,
                        mllm_eas_model_name,
                        mllm_api_model_name,
                    ]
                )

                def choose_use_mllm(value):
                    if value:
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)

                use_mllm.change(
                    fn=choose_use_mllm,
                    inputs=use_mllm,
                    outputs=use_mllm_col,
                )

                def change_llm(value):
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

                llm.input(
                    fn=change_llm,
                    inputs=llm,
                    outputs=[eas_col, api_llm_col, llm_api_model_name],
                )

                def change_mllm(value):
                    eas_visible = value == "PaiEas"
                    api_visible = value != "PaiEas"
                    model_options = MLLM_MODEL_KEY_DICT.get(value, [])
                    cur_model = model_options[0] if model_options else ""
                    return {
                        m_eas_col: gr.update(visible=eas_visible),
                        api_mllm_col: gr.update(visible=api_visible),
                        mllm_api_model_name: gr.update(
                            choices=model_options, value=cur_model
                        ),
                    }

                mllm.input(
                    fn=change_mllm,
                    inputs=mllm,
                    outputs=[m_eas_col, api_mllm_col, mllm_api_model_name],
                )

        vector_db_elems = create_vector_db_panel(
            input_elements={
                llm,
                llm_eas_url,
                llm_eas_token,
                llm_eas_model_name,
                mllm,
                mllm_eas_url,
                mllm_eas_token,
                mllm_eas_model_name,
                embed_source,
                embed_model,
                embed_dim,
                embed_batch_size,
                llm_api_model_name,
                mllm_api_model_name,
            },
            connect_vector_func=connect_vector_db,
        )
    with gr.Row():
        _ = gr.Button("Save", variant="primary")
    elems = components_to_dict(components)
    elems.update(vector_db_elems)
    return elems
