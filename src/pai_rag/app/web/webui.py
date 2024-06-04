import gradio as gr
from pai_rag.app.web.view_model import view_model
from pai_rag.app.web.tabs.settings_tab import create_setting_tab
from pai_rag.app.web.tabs.upload_tab import create_upload_tab
from pai_rag.app.web.tabs.chat_tab import create_chat_tab
from pai_rag.app.web.element_manager import elem_manager
from pai_rag.app.web.ui_constants import (
    DEFAULT_CSS_STYPE,
    WELCOME_MESSAGE,
)

import logging

logger = logging.getLogger("WebUILogger")


def resume_ui():
    outputs = {}
    component_settings = view_model.to_component_settings()

    for elem in elem_manager.get_elem_list():
        elem_id = elem.elem_id
        elem_attr = component_settings[elem_id]
        elem = elem_manager.get_elem_by_id(elem_id=elem_id)

        # For gradio version 3.41.0, we can remove .value for latest gradio here.
        outputs[elem] = elem.__class__(**elem_attr).value

    return outputs


def create_ui():
    with gr.Blocks(css=DEFAULT_CSS_STYPE) as homepage:
        # generate components
        gr.Markdown(value=WELCOME_MESSAGE)
        with gr.Tab("\N{rocket} Settings"):
            setting_elements = create_setting_tab()
            elem_manager.add_elems(setting_elements)
            print("load settings complete ===")
        with gr.Tab("\N{whale} Upload"):
            upload_elements = create_upload_tab()
            elem_manager.add_elems(upload_elements)
            print("load upload complete ===")

        with gr.Tab("\N{fire} Chat"):
            chat_elements = create_chat_tab()
            elem_manager.add_elems(chat_elements)
        homepage.load(
            resume_ui, outputs=elem_manager.get_elem_list(), concurrency_limit=None
        )
        print("load web ui complete ===")
    return homepage
