from typing import Dict, Any
import gradio as gr
from pai_rag.app.web.ui_constants import EMBEDDING_API_KEY_DICT
from pai_rag.app.web.utils import components_to_dict
from pai_rag.app.web.tabs.vector_db_panel import create_vector_db_panel
import logging
import os
import pai_rag.app.web.event_listeners as ev_listeners

logger = logging.getLogger(__name__)

DEFAULT_IS_INTERACTIVE = os.environ.get("PAIRAG_RAG__SETTING__interactive", "true")


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
                    embed_type = gr.Textbox(
                        label="Embedding Type",
                        elem_id="embed_type",
                    )
                    embed_link = gr.Markdown(
                        label="Model URL Link",
                        elem_id="embed_link",
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
                    oss_endpoint = gr.Textbox(
                        label="OSS Endpoint",
                        elem_id="oss_endpoint",
                    )
                    oss_bucket = gr.Textbox(
                        label="OSS Bucket",
                        elem_id="oss_bucket",
                    )

            embed_source.input(
                fn=ev_listeners.change_emb_source,
                inputs=[embed_source, embed_model],
                outputs=[embed_model, embed_dim, embed_type, embed_link],
            )
            embed_model.input(
                fn=ev_listeners.change_emb_model,
                inputs=[embed_source, embed_model],
                outputs=[embed_dim, embed_type, embed_link],
            )
            use_oss.change(
                fn=ev_listeners.change_use_oss,
                inputs=use_oss,
                outputs=oss_col,
            )
            components.extend(
                [
                    embed_source,
                    embed_dim,
                    embed_type,
                    embed_link,
                    embed_model,
                    embed_batch_size,
                    use_oss,
                    oss_ak,
                    oss_sk,
                    oss_endpoint,
                    oss_bucket,
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
                    with gr.Column(visible=(mllm == "DashScope")) as api_mllm_col:
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

                use_mllm.change(
                    fn=ev_listeners.choose_use_mllm,
                    inputs=use_mllm,
                    outputs=[use_mllm_col],
                )

                llm.change(
                    fn=ev_listeners.change_llm,
                    inputs=llm,
                    outputs=[
                        llm_eas_url,
                        llm_eas_token,
                        llm_eas_model_name,
                        llm_api_model_name,
                    ],
                )

                mllm.change(
                    fn=ev_listeners.change_mllm,
                    inputs=mllm,
                    outputs=[m_eas_col, api_mllm_col, mllm_api_model_name],
                )

        vector_db_elems, vector_db_components = create_vector_db_panel(
            input_elements={
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
                embed_source,
                embed_model,
                embed_dim,
                embed_type,
                embed_link,
                embed_batch_size,
                use_oss,
                oss_ak,
                oss_sk,
                oss_endpoint,
                oss_bucket,
            }
        )
    save_btn = gr.Button("Save", variant="primary")
    save_state = gr.Textbox(label="Connection Info: ", container=False)
    save_btn.click(
        fn=ev_listeners.save_config,
        inputs=vector_db_elems,
        outputs=[oss_ak, oss_sk, save_state],
        api_name="save_config",
    )
    elems = components_to_dict(components)
    elems.update(vector_db_components)
    return elems
