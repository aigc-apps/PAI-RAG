from typing import Dict, Any
import gradio as gr
from pai_rag.app.web.ui_constants import EMBEDDING_API_KEY_DICT
from pai_rag.app.web.utils import components_to_dict
from pai_rag.app.web.index_utils import index_related_component_keys
from pai_rag.app.web.tabs.vector_db_panel import create_vector_db_panel
import logging
import os
import pai_rag.app.web.event_listeners as ev_listeners

logger = logging.getLogger(__name__)

DEFAULT_IS_INTERACTIVE = os.environ.get("PAIRAG_RAG__SETTING__interactive", "true")


def create_setting_tab() -> Dict[str, Any]:
    components = []
    with gr.Row():
        with gr.Column(variant="panel"):
            with gr.Column(scale=5):
                _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **Index**")

                vector_index = gr.Dropdown(
                    label="Index Name",
                    choices=["NEW"],
                    value="NEW",
                    interactive=True,
                    elem_id="vector_index",
                )

                new_index_name = gr.Textbox(
                    label="New Index Name",
                    value="",
                    interactive=True,
                    elem_id="new_index_name",
                    visible=False,
                )

                # _ = gr.Markdown(value="**Index - Embedding Model**")
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
            vector_db_elems, vector_db_components = create_vector_db_panel()

            add_index_button = gr.Button(
                "Add Index",
                variant="primary",
                visible=False,
                elem_id="add_index_button",
            )
            update_index_button = gr.Button(
                "Update Index",
                variant="primary",
                visible=False,
                elem_id="update_index_button",
            )
            delete_index_button = gr.Button(
                "Delete Index",
                variant="stop",
                visible=False,
                elem_id="delete_index_button",
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
            components.extend(
                [
                    embed_source,
                    embed_dim,
                    embed_type,
                    embed_link,
                    embed_model,
                    embed_batch_size,
                    vector_index,
                    new_index_name,
                    add_index_button,
                    update_index_button,
                    delete_index_button,
                ]
            )

            all_component = {element.elem_id: element for element in vector_db_elems}
            all_component.update(
                {component.elem_id: component for component in components}
            )
            index_related_components = [
                all_component[key] for key in index_related_component_keys
            ]
            add_index_button.click(
                fn=ev_listeners.add_index,
                inputs=index_related_components,
                outputs=[
                    vector_index,
                    new_index_name,
                    add_index_button,
                    update_index_button,
                    delete_index_button,
                ],
            )

            update_index_button.click(
                fn=ev_listeners.update_index,
                inputs=index_related_components,
                outputs=[
                    vector_index,
                    new_index_name,
                    add_index_button,
                    update_index_button,
                    delete_index_button,
                ],
            )

            """
            delete_index_button.click(
                fn=ev_listeners.delete_index,
                inputs=[vector_index],
                outputs=[],
                visible=False,
            )
            """

        with gr.Column(variant="panel"):
            with gr.Row():
                _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **Large Language Model**")
                llm = gr.Radio(
                    ["paieas", "dashscope", "openai"],
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
            with gr.Column(variant="panel"):
                _ = gr.Markdown(
                    value="\N{WHITE MEDIUM STAR} **(Optional) Multi-Modal Large Language Model**"
                )
                use_mllm = gr.Checkbox(
                    label="Use Multi-Modal LLM",
                    elem_id="use_mllm",
                    container=False,
                )
                with gr.Row(visible=False, elem_id="use_mllm_col") as use_mllm_col:
                    mllm = gr.Radio(
                        ["paieas", "dashscope"],
                        label="LLM Model Source",
                        elem_id="mllm",
                        interactive=DEFAULT_IS_INTERACTIVE.lower() != "false",
                    )
                    with gr.Row(
                        visible=(mllm == "paieas"), elem_id="m_eas_col"
                    ) as m_eas_col:
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
                    with gr.Row(
                        visible=(mllm == "dashscope"), elem_id="api_mllm_col"
                    ) as api_mllm_col:
                        mllm_api_model_name = gr.Dropdown(
                            label="LLM Model Name",
                            elem_id="mllm_api_model_name",
                        )

            with gr.Column(scale=5, variant="panel"):
                _ = gr.Markdown(
                    value="\N{WHITE MEDIUM STAR} **(Optional, for saving image & load data) OSS Bucket**"
                )
                use_oss = gr.Checkbox(
                    label="Use OSS Storage",
                    elem_id="use_oss",
                    container=False,
                )
                with gr.Row(visible=False, elem_id="use_oss_col") as use_oss_col:
                    oss_bucket = gr.Textbox(
                        label="OSS Bucket",
                        elem_id="oss_bucket",
                    )
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
                        default="oss-cn-hangzhou.aliyuncs.com",
                    )
                use_oss.input(
                    fn=ev_listeners.change_use_oss,
                    inputs=use_oss,
                    outputs=use_oss_col,
                )

            llm_components = [
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
                use_oss,
                oss_ak,
                oss_sk,
                oss_endpoint,
                oss_bucket,
            ]

            components.extend(llm_components)

            use_mllm.input(
                fn=ev_listeners.choose_use_mllm,
                inputs=use_mllm,
                outputs=[use_mllm_col],
            )

            llm.input(
                fn=ev_listeners.change_llm,
                inputs=llm,
                outputs=[
                    llm_eas_url,
                    llm_eas_token,
                    llm_eas_model_name,
                    llm_api_model_name,
                ],
            )

            mllm.input(
                fn=ev_listeners.change_mllm,
                inputs=mllm,
                outputs=[m_eas_col, api_mllm_col, mllm_api_model_name],
            )

            save_btn = gr.Button("Save Llm Setting", variant="primary")
            save_state = gr.Textbox(label="Connection Info: ", container=False)
            save_btn.click(
                fn=ev_listeners.save_config,
                inputs=set(llm_components),
                outputs=[oss_ak, oss_sk, save_state],
                api_name="save_config",
            )
    elems = components_to_dict(components)
    elems.update(vector_db_components)
    elems.update(
        {
            m_eas_col.elem_id: m_eas_col,
            api_mllm_col.elem_id: api_mllm_col,
            use_oss_col.elem_id: use_oss_col,
            use_mllm_col.elem_id: use_mllm_col,
        }
    )
    return elems
