from typing import Dict, Any
import gradio as gr
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


def create_search_web_tab() -> Dict[str, Any]:
    with gr.Column(visible=True):
        search_type = gr.Radio(
            ["bing", "aliyun", "google"],
            label="搜索引擎",
            elem_id="search_type",
        )
        serpapi_key_tips = gr.Markdown(value="如何获取 [SerpAPI Key](https://serpapi.com)")
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
        aliyun_endpoint = gr.Text(label="Endpoint", value="", elem_id="aliyun_endpoint")
        aliyun_access_key_id = gr.Text(
            label="AccessKey ID", value="", elem_id="aliyun_access_key_id"
        )
        aliyun_access_key_secret = gr.Text(
            label="AccessKey Secret",
            value="",
            type="password",
            elem_id="aliyun_access_key_secret",
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

    elems = components_to_dict(components)
    return elems
