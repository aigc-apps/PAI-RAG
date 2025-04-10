import datetime
import gradio as gr
from typing import Dict, Any, List
from pai_rag.app.web.rag_local_client import RagApiError, rag_client
from pai_rag.app.web.utils import components_to_dict


def change_search_model_argument(search_type):
    return [
        gr.update(visible=True if search_type == "bing" else False),
        gr.update(visible=True),
        gr.update(visible=True if search_type in ["bing", "google"] else False),
        gr.update(visible=True if search_type == "aliyun" else False),
        gr.update(visible=True if search_type == "aliyun" else False),
        gr.update(visible=True if search_type == "aliyun" else False),
        gr.update(visible=True if search_type == "google" else False),
        gr.update(visible=True if search_type == "google" else False),
    ]


def save_search_web_cfg_func(input_elements: List[Any]):
    try:
        update_dict = {}
        for element, value in input_elements.items():
            update_dict[element.elem_id] = value
        rag_client.patch_config(update_dict)

        return gr.update(
            value=f"[{datetime.datetime.now()}] Web Search configuration saved successfully!",
            visible=True,
        )
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def create_search_web_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(visible=True, scale=3):
            _ = gr.Markdown(value="## \N{WHITE MEDIUM STAR} **配置搜索引擎**")
            search_type = gr.Radio(
                ["bing", "aliyun", "google"],
                label="搜索引擎",
                elem_id="search_type",
            )
            serpapi_key_tips = gr.Markdown(
                value="如何获取 [SerpAPI Key](https://serpapi.com)"
            )
            search_api_key = gr.Text(
                label="Bing API Key",
                value="",
                type="password",
                elem_id="search_api_key",
            )
            serpapi_key = gr.Text(
                label="SerpAPI Key",
                value="",
                type="password",
                elem_id="serpapi_key",
            )
            search_count = gr.Slider(
                label="搜索数量",
                minimum=5,
                maximum=50,
                step=1,
                elem_id="search_count",
            )
            search_lang = gr.Radio(
                label="语言",
                choices=["zh-CN", "en-US"],
                value="zh-CN",
                elem_id="search_lang",
            )
            aliyun_endpoint = gr.Text(
                label="Endpoint", value="", elem_id="aliyun_endpoint"
            )
            aliyun_access_key_id = gr.Text(
                label="AccessKey ID", value="", elem_id="aliyun_access_key_id"
            )
            aliyun_access_key_secret = gr.Text(
                label="AccessKey Secret",
                value="",
                type="password",
                elem_id="aliyun_access_key_secret",
            )
        with gr.Column(scale=7):
            _ = gr.Markdown(value="## \N{WHITE MEDIUM STAR} **调整提示词模板**")
            search_qa_prompt_template = gr.Textbox(
                label="联网搜索问答的提示词模板",
                value="",
                elem_id="search_qa_prompt_template",
                lines=10,
                interactive=True,
            )
    with gr.Row():
        with gr.Column():
            save_search_web_cfg = gr.Button(
                value="检查并保存配置",
                elem_id="save_search_web_cfg",
                variant="primary",
            )
            save_state_for_search_web = gr.Textbox(
                label="Save Info: ", container=False, visible=True
            )
        components = [
            search_type,
            search_api_key,
            search_count,
            search_lang,
            aliyun_endpoint,
            aliyun_access_key_id,
            aliyun_access_key_secret,
            serpapi_key,
            search_qa_prompt_template,
            save_search_web_cfg,
            save_state_for_search_web,
        ]
        search_type.input(
            fn=change_search_model_argument,
            inputs=[search_type],
            outputs=[
                search_api_key,
                search_count,
                search_lang,
                aliyun_endpoint,
                aliyun_access_key_id,
                aliyun_access_key_secret,
                serpapi_key,
                serpapi_key_tips,
            ],
        )
        save_search_web_cfg.click(
            fn=save_search_web_cfg_func,
            inputs=set(components),
            outputs=[save_state_for_search_web],
            api_name="save_search_web_cfg",
        )

    elems = components_to_dict(components)
    return elems
