from fastapi import FastAPI
import gradio as gr
from pai_rag.app.web import event_listeners
from pai_rag.app.web.index_utils import index_to_components_settings
from pai_rag.app.web.tabs.agent_tab import create_agent_tab
from pai_rag.app.web.view_model import ViewModel
from pai_rag.app.web.rag_local_client import rag_client
from pai_rag.app.web.tabs.settings_tab import create_setting_tab
from pai_rag.app.web.tabs.chat_tab import create_chat_tab
from pai_rag.app.web.tabs.data_analysis_tab import create_data_analysis_tab
from pai_rag.app.web.tabs.history_tab import (
    create_upload_history,
    refresh_upload_history,
)
from pai_rag.app.web.index_utils import index_related_component_keys
from pai_rag.knowledgebase.rag_knowledgebase import KnowledgeBase
from pai_rag.utils.constants import DEFAULT_KNOWLEDGEBASE_NAME

# from pai_rag.app.web.tabs.eval_tab import create_evaluation_tab
from pai_rag.app.web.element_manager import elem_manager
from pai_rag.app.web.ui_constants import (
    DEFAULT_CSS_STYPE,
    WELCOME_MESSAGE,
)
from pai_rag.app.web.tabs.model.index_info import get_index_map
from pai_rag.app.web.filebrowser.request_utils import postprocess_middleware


def resume_ui():
    outputs = {}
    rag_config = rag_client.get_config()
    view_model = ViewModel.from_app_config(rag_config)
    index_map = get_index_map()
    component_settings = view_model.to_component_settings()
    default_index = KnowledgeBase(
        name=DEFAULT_KNOWLEDGEBASE_NAME,
        vector_store_config=rag_config.index.vector_store,
        embedding_config=rag_config.embedding,
    )
    component_settings.update(
        index_to_components_settings(
            default_index, index_list=list(index_map.knowledgebases.keys())
        )
    )

    upload_summary, upload_history = refresh_upload_history(DEFAULT_KNOWLEDGEBASE_NAME)
    outputs[elem_manager.get_elem_by_id("upload_summary")] = upload_summary
    outputs[elem_manager.get_elem_by_id("upload_history")] = upload_history

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


def change_chat_page_model_list(model_id):
    rag_config = rag_client.get_config()
    model_choices = [
        llm.model_id if llm.model_id else llm.model for llm in rag_config.llms
    ]
    if rag_config.chat.model_id and rag_config.chat.model_id in model_choices:
        new_model_id = rag_config.chat.model_id
    else:
        new_model_id = model_choices[0]

    if (
        rag_config.data_analysis.model_id
        and rag_config.data_analysis.model_id in model_choices
    ):
        data_analysis_model_id = rag_config.data_analysis.model_id
    else:
        data_analysis_model_id = model_choices[0]
    return [
        gr.update(choices=model_choices, value=new_model_id),
        gr.update(choices=model_choices, value=new_model_id),
        gr.update(choices=model_choices, value=data_analysis_model_id),
    ]


def change_vector_index_button(index_name):
    if index_name == "NEW":
        return [
            gr.update(),
            gr.update(),
            gr.update(),
        ]
    index_map = get_index_map()
    index_list = list(index_map.knowledgebases.keys())
    return [
        gr.update(choices=index_list + ["NEW"], value=index_name),
        gr.update(choices=index_list, value=index_name),
        gr.update(choices=index_list, value=index_name),
    ]


def make_homepage():
    with gr.Blocks(css=DEFAULT_CSS_STYPE) as homepage:
        # generate components
        gr.Markdown(value=WELCOME_MESSAGE)
        with gr.Tab("\N{rocket} 系统设置"):
            setting_elements = create_setting_tab()
            elem_manager.add_elems(setting_elements)
        # with gr.Tab("\N{whale} Upload"):
        #     upload_elements = create_upload_tab()
        #     elem_manager.add_elems(upload_elements)
        with gr.Tab("\N{fire} 对话"):
            chat_elements = create_chat_tab()
            elem_manager.add_elems(chat_elements)
        with gr.Tab("\N{rocket} 智能体"):
            agent_elements = create_agent_tab()
            elem_manager.add_elems(agent_elements)
        with gr.Tab("\N{bar chart} 数据分析"):
            analysis_elements = create_data_analysis_tab()
            elem_manager.add_elems(analysis_elements)
        with gr.Tab("\N{rocket} 知识库"):
            with gr.Tab("文件管理"):
                with gr.Blocks():
                    html = '<iframe src="./filebrowser" width="100%" height="1000" title="FileBrowser"></iframe>'
                    gr.HTML(html)
            with gr.Tab("上传历史"):
                history_elements = create_upload_history()
                elem_manager.add_elems(history_elements)

        index_selector_elements = [
            setting_elements["vector_index"],
            # upload_elements["upload_index"],
            chat_elements["chat_index"],
            history_elements["history_index"],
        ]
        index_related_components = [
            setting_elements[key] for key in index_related_component_keys
        ]

        setting_elements["vector_index"].change(
            event_listeners.change_vector_index,
            inputs=setting_elements["vector_index"],
            outputs=index_related_components
            + [chat_elements["chat_index"], history_elements["history_index"]],
        )
        history_elements["history_index"].input(
            change_vector_index_button,
            inputs=history_elements["history_index"],
            outputs=index_selector_elements,
        )
        chat_elements["chat_index"].input(
            change_vector_index_button,
            inputs=chat_elements["chat_index"],
            outputs=index_selector_elements,
        )

        setting_elements["llm_model"].change(
            change_chat_page_model_list,
            inputs=setting_elements["llm_model"],
            outputs=[
                chat_elements["chat_model_id"],
                chat_elements["query_rewrite_model_id"],
                analysis_elements["data_analysis_model_id"],
            ],
        )

        # with gr.Tab("\N{rocket} Evaluation"):
        #     eval_elements = create_evaluation_tab()
        #     elem_manager.add_elems(eval_elements)
        homepage.load(
            resume_ui, outputs=elem_manager.get_elem_list(), concurrency_limit=None
        )
    return homepage


def configure_webapp(app: FastAPI) -> gr.Blocks:
    # 添加中间件
    app.middleware("http")(postprocess_middleware)
    home = make_homepage()
    gr.mount_gradio_app(app, home, path="/")
    return
