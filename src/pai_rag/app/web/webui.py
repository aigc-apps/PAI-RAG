from fastapi import FastAPI
import gradio as gr
import os
from pai_rag.app.web import event_listeners
from pai_rag.app.web.index_utils import index_to_components_settings
from pai_rag.app.web.tabs.agent_tab import create_agent_tab
from pai_rag.app.web.view_model import ViewModel
from pai_rag.app.web.rag_client import DEFAULT_LOCAL_URL, rag_client
from pai_rag.app.web.tabs.settings_tab import create_setting_tab
from pai_rag.app.web.tabs.upload_tab import create_upload_tab
from pai_rag.app.web.tabs.chat_tab import create_chat_tab
from pai_rag.app.web.tabs.data_analysis_tab import create_data_analysis_tab
from pai_rag.app.web.index_utils import index_related_component_keys

# from pai_rag.app.web.tabs.eval_tab import create_evaluation_tab
from pai_rag.app.web.element_manager import elem_manager
from pai_rag.app.web.ui_constants import (
    DEFAULT_CSS_STYPE,
    WELCOME_MESSAGE,
)
from pai_rag.app.web.tabs.model.index_info import get_index_map

import logging

DEFAULT_IS_INTERACTIVE = os.environ.get("PAIRAG_RAG__SETTING__interactive", "true")

logger = logging.getLogger("WebUILogger")


def resume_ui():
    outputs = {}
    rag_config = rag_client.get_config()
    view_model = ViewModel.from_app_config(rag_config)
    index_map = get_index_map()
    component_settings = view_model.to_component_settings()
    default_index = index_map.indexes[index_map.current_index_name]
    component_settings.update(
        index_to_components_settings(
            default_index, index_list=list(index_map.indexes.keys())
        )
    )

    for elem in elem_manager.get_elem_list():
        elem_id = elem.elem_id
        if elem_id in component_settings.keys():
            elem_attr = component_settings[elem_id]
            elem = elem_manager.get_elem_by_id(elem_id=elem_id)
            # For gradio version 3.41.0, we can remove .value for latest gradio here.
            outputs[elem] = gr.update(**elem_attr)
            # if elem_id == "qa_dataset_file":
            #     outputs[elem] = elem_attr["value"]
            # else:
            #     outputs[elem] = elem.__class__(**elem_attr).value

    return outputs


def change_vector_index_button(index_name):
    if index_name == "NEW":
        return [
            gr.update(),
            gr.update(),
            gr.update(),
        ]
    index_map = get_index_map()
    index_list = list(index_map.indexes.keys())
    return [
        gr.update(choices=index_list + ["NEW"], value=index_name),
        gr.update(choices=index_list, value=index_name),
        gr.update(choices=index_list, value=index_name),
    ]


def make_homepage():
    with gr.Blocks(css=DEFAULT_CSS_STYPE) as homepage:
        # generate components
        gr.Markdown(value=WELCOME_MESSAGE)
        with gr.Tab("\N{rocket} Settings"):
            setting_elements = create_setting_tab()
            elem_manager.add_elems(setting_elements)
        with gr.Tab("\N{whale} Upload"):
            upload_elements = create_upload_tab()
            elem_manager.add_elems(upload_elements)
        with gr.Tab("\N{fire} Chat"):
            chat_elements = create_chat_tab()
            elem_manager.add_elems(chat_elements)
        with gr.Tab("\N{rocket} Agent"):
            agent_elements = create_agent_tab()
            elem_manager.add_elems(agent_elements)
        with gr.Tab("\N{bar chart} Data Analysis"):
            analysis_elements = create_data_analysis_tab()
            elem_manager.add_elems(analysis_elements)

        index_selector_elements = [
            setting_elements["vector_index"],
            upload_elements["upload_index"],
            chat_elements["chat_index"],
        ]
        index_related_components = [
            setting_elements[key] for key in index_related_component_keys
        ]

        setting_elements["vector_index"].change(
            event_listeners.change_vector_index,
            inputs=setting_elements["vector_index"],
            outputs=index_related_components
            + [upload_elements["upload_index"], chat_elements["chat_index"]],
        )
        upload_elements["upload_index"].input(
            change_vector_index_button,
            inputs=upload_elements["upload_index"],
            outputs=index_selector_elements,
        )
        chat_elements["chat_index"].input(
            change_vector_index_button,
            inputs=chat_elements["chat_index"],
            outputs=index_selector_elements,
        )

        # with gr.Tab("\N{rocket} Evaluation"):
        #     eval_elements = create_evaluation_tab()
        #     elem_manager.add_elems(eval_elements)
        homepage.load(
            resume_ui, outputs=elem_manager.get_elem_list(), concurrency_limit=None
        )
    return homepage


def configure_webapp(app: FastAPI, web_url, rag_url=DEFAULT_LOCAL_URL) -> gr.Blocks:
    rag_client.set_endpoint(rag_url)
    home = make_homepage()
    home.queue(concurrency_count=1, max_size=64)
    home._queue.set_url(web_url)
    print(web_url)
    gr.mount_gradio_app(app, home, path="")
    return home
