from typing import Dict, Any, List
import gradio as gr
from pai_rag.web.ui_constants import EMBEDDING_API_KEY_DICT
from pai_rag.web.utils import components_to_dict
from pai_rag.web.index_utils import index_related_component_keys
from pai_rag.web.tabs.vector_db_panel import create_vector_db_panel
import pai_rag.web.event_listeners as ev_listeners
from pai_rag.web.rag_local_client import RagApiError, rag_client
from pai_rag.web.tabs.history_tab import create_upload_history
from pai_rag.web.tabs.chat_tab import reset_textbox, clear_history
from pai_rag.web.view_model import (
    RETRIEVAL_MODE_MAP,
    INVERTED_RETRIEVAL_MODE_MAP,
    RERANKER_TYPE_MAP,
    INVERTED_RERANKER_TYPE_MAP,
)
from loguru import logger
import json
import datetime


async def retrieval_test_respond(input_elements: List[Any]):
    update_dict = {}
    for element, value in input_elements.items():
        update_dict[element.elem_id] = value

    # empty input.
    if not update_dict["retrieval_test_question"]:
        yield update_dict["retrieval_test_chatbot"]
        return

    chatbot = update_dict["retrieval_test_chatbot"]
    question = update_dict["retrieval_test_question"]
    q_msg = {"content": question, "role": "user"}
    chatbot.append(q_msg)
    index_name = update_dict["retrieval_test_chat_index"]

    if chatbot is not None:
        chatbot.append(
            {"content": "", "role": "assistant", "metadata": {"status": "pending"}}
        )
        yield chatbot

    try:
        print(
            'QUERY_TYPE_MAP.get(update_dict["retrieval_mode"])',
            RETRIEVAL_MODE_MAP.get(update_dict["retrieval_mode"]),
            update_dict["retrieval_mode"],
        )
        response_gen = rag_client.aknowledgebase_retrieval(
            knowledgebase_id=index_name,
            query=question,
            retrieval_settings={
                "retrieval_mode": RETRIEVAL_MODE_MAP.get(update_dict["retrieval_mode"]),
                "similarity_top_k": update_dict["similarity_top_k"],
                # "image_similarity_top_k": update_dict["image_similarity_top_k"], # not supported yet
                "reranker_type": RERANKER_TYPE_MAP.get(update_dict["reranker_type"]),
                "similarity_threshold": update_dict["similarity_threshold"],
                "reranker_similarity_threshold": update_dict[
                    "reranker_similarity_threshold"
                ],
                "reranker_model": update_dict["reranker_model"],
                "reranker_similarity_top_k": update_dict["reranker_similarity_top_k"],
            },
        )

        async for resp in response_gen:
            chatbot[-1]["content"] += resp.delta
            yield chatbot

    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")
    except Exception as e:
        raise gr.Error(f"Error: {e}")
    finally:
        logger.info(f"Chatbot finished: {chatbot}")
        yield chatbot


def save_retrieval_config(input_elements: List[Any]):
    update_dict = {}
    for element, value in input_elements.items():
        update_dict[element.elem_id] = value
    knowledgebase_id = update_dict["retrieval_test_chat_index"]
    retrieval_settings = {
        "retrieval_mode": RETRIEVAL_MODE_MAP.get(update_dict["retrieval_mode"]),
        "similarity_top_k": update_dict["similarity_top_k"],
        "reranker_type": RERANKER_TYPE_MAP.get(update_dict["reranker_type"]),
        "similarity_threshold": update_dict["similarity_threshold"],
        "reranker_similarity_threshold": update_dict["reranker_similarity_threshold"],
        "reranker_model": update_dict["reranker_model"],
        "reranker_similarity_top_k": update_dict["reranker_similarity_top_k"],
    }
    rag_client.update_index_retrieval_settings(knowledgebase_id, retrieval_settings)
    index_retrieval_settings = {
        "knowledgebase_id": knowledgebase_id,
        "retrieval_settings": retrieval_settings,
    }
    return json.dumps(index_retrieval_settings, indent=4, ensure_ascii=False)


def save_knowledgebase_qa_prompt_func(input_elements: List[Any]):
    update_dict = {}
    for element, value in input_elements.items():
        update_dict[element.elem_id] = value
    knowledgebase_id = update_dict["knowledgebase_qa_prompt_index"]
    qa_prompt_templates = {
        "system_prompt_template": update_dict[
            "knowledgebase_qa_system_prompt_template"
        ],
        "task_prompt_template": update_dict["knowledgebase_qa_task_prompt_template"],
    }
    rag_client.update_index_qa_prompt_templates(knowledgebase_id, qa_prompt_templates)
    return gr.update(
        value=f"[{datetime.datetime.now()}] QA prompt templated for knowledgebase {knowledgebase_id} saved successfully!",
        visible=True,
    )


def show_knowledgebase_qa_prompt_templates(knowledgebase_qa_prompt_index):
    qa_prompt_templates = rag_client.get_index_qa_prompt_templates(
        knowledgebase_qa_prompt_index
    )
    return [
        gr.update(value=qa_prompt_templates["system_prompt_template"]),
        gr.update(value=qa_prompt_templates["task_prompt_template"]),
    ]


def show_retrieval_config(retrieval_test_chat_index):
    retrieval_settings = rag_client.get_index_retrieval_settings(
        retrieval_test_chat_index
    )
    index_retrieval_settings = {
        "knowledgebase_id": retrieval_test_chat_index,
        "retrieval_settings": retrieval_settings,
    }

    return [
        json.dumps(index_retrieval_settings, indent=4, ensure_ascii=False),
        gr.update(
            value=INVERTED_RETRIEVAL_MODE_MAP.get(retrieval_settings["retrieval_mode"])
        ),
        gr.update(
            value=INVERTED_RERANKER_TYPE_MAP.get(retrieval_settings["reranker_type"])
        ),
        gr.update(value=retrieval_settings["similarity_top_k"]),
        gr.update(value=retrieval_settings["similarity_threshold"]),
        gr.update(value=retrieval_settings["reranker_similarity_threshold"]),
        gr.update(value=retrieval_settings["reranker_model"]),
        gr.update(value=retrieval_settings["reranker_similarity_top_k"]),
    ]


def create_knowledgebase_settings_tab() -> Dict[str, Any]:
    components = []
    with gr.Row(variant="panel"):
        with gr.Column(scale=5):
            _ = gr.Markdown(value="### **知识库**")

            vector_index = gr.Dropdown(
                label="知识库名称",
                choices=["NEW"],
                value="NEW",
                interactive=True,
                elem_id="vector_index",
                allow_custom_value=True,
            )

            new_index_name = gr.Textbox(
                label="新知识库名称",
                value="",
                interactive=True,
                elem_id="new_index_name",
                visible=False,
            )

            _ = gr.Markdown(value="**知识库 - 向量模型**")
            embed_source = gr.Radio(
                EMBEDDING_API_KEY_DICT.keys(),
                label="向量模型来源",
                elem_id="embed_source",
                interactive=True,
            )
            embed_model = gr.Dropdown(
                label="向量模型名称",
                elem_id="embed_model",
                visible=False,
            )
            with gr.Row():
                embed_dim = gr.Textbox(
                    label="向量维度",
                    elem_id="embed_dim",
                )
                embed_batch_size = gr.Textbox(
                    label="向量Batch大小",
                    elem_id="embed_batch_size",
                )
                embed_type = gr.Textbox(
                    label="向量模型类型",
                    elem_id="embed_type",
                )
                embed_api_key = gr.Textbox(
                    label="API KEY",
                    elem_id="embed_api_key",
                    visible=False,
                    type="password",
                )
        with gr.Column(scale=5):
            vector_db_elems, vector_db_components = create_vector_db_panel()
            with gr.Row():
                with gr.Column():
                    _ = gr.Markdown(value="**切片配置**")

                    chunk_size = gr.Textbox(
                        label="\N{rocket} 块大小（文档被分割成的块的大小）",
                        elem_id="chunk_size",
                        interactive=True,
                    )

                    chunk_overlap = gr.Textbox(
                        label="\N{fire} 块重叠（相邻文档块之间相互重叠的部分）",
                        elem_id="chunk_overlap",
                        interactive=True,
                    )

    with gr.Row():
        add_index_button = gr.Button(
            "添加知识库",
            variant="primary",
            visible=False,
            elem_id="add_index_button",
        )
        update_index_button = gr.Button(
            "更新知识库",
            variant="primary",
            visible=False,
            elem_id="update_index_button",
        )
        delete_index_button = gr.Button(
            "删除知识库",
            variant="stop",
            visible=False,
            elem_id="delete_index_button",
        )

        embed_source.input(
            fn=ev_listeners.change_emb_source,
            inputs=[embed_source, embed_model],
            outputs=[embed_model, embed_dim, embed_type, embed_api_key],
        )
        embed_model.input(
            fn=ev_listeners.change_emb_model,
            inputs=[embed_source, embed_model],
            outputs=[embed_dim, embed_type],
        )
        components.extend(
            [
                embed_source,
                embed_dim,
                embed_type,
                embed_model,
                embed_api_key,
                embed_batch_size,
                vector_index,
                new_index_name,
                add_index_button,
                update_index_button,
                delete_index_button,
                chunk_size,
                chunk_overlap,
            ]
        )

        all_elements = {element.elem_id: element for element in vector_db_elems}
        all_elements.update({component.elem_id: component for component in components})

        index_related_elements = [
            all_elements[key] for key in index_related_component_keys
        ]
        add_index_button.click(
            fn=ev_listeners.add_index,
            inputs=index_related_elements,
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
            inputs=index_related_elements,
            outputs=[
                vector_index,
                new_index_name,
                add_index_button,
                update_index_button,
                delete_index_button,
            ],
        )

        delete_index_button.click(
            fn=ev_listeners.delete_index,
            inputs=[vector_index],
            outputs=[
                vector_index,
                new_index_name,
                add_index_button,
                update_index_button,
                delete_index_button,
            ],
        )

    return all_elements


def create_retrieval_test_tab():
    components = []
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Column():
                _ = gr.Markdown(value="### **检索参数调试**")
                retrieval_mode = gr.Radio(
                    ["向量检索", "关键字检索", "混合检索"],
                    label="检索模式",
                    elem_id="retrieval_mode",
                )

                similarity_top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    elem_id="similarity_top_k",
                    label="返回Top-K条文本结果 (0 到 100)",
                )
                # image_similarity_top_k = gr.Slider(
                #     minimum=0,
                #     maximum=10,
                #     step=1,
                #     elem_id="image_similarity_top_k",
                #     label="返回Top-K条图片结果 (0 到 10)",
                # )
                similarity_threshold = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    elem_id="similarity_threshold",
                    label="相似度分数阈值 (内容越相似，分数越大)",
                )

                reranker_type = gr.Radio(
                    ["无重排序", "基于模型的重排序"],
                    label="重排序类型",
                    elem_id="reranker_type",
                )
                with gr.Column(
                    visible=(reranker_type == "基于模型的重排序"),
                    elem_id="model_reranker_col",
                ) as model_reranker_col:
                    reranker_model = gr.Radio(
                        [
                            "bge-reranker-base",
                            "bge-reranker-large",
                        ],
                        label="重排序模型（注意：首次使用该模型时，加载模型将需要较长时间）",
                        elem_id="reranker_model",
                    )
                    reranker_similarity_threshold = gr.Slider(
                        minimum=-10,
                        maximum=10,
                        step=0.01,
                        elem_id="reranker_similarity_threshold",
                        label="重排序相似度分数阈值（结果越相似，数值越大）",
                    )
                    reranker_similarity_top_k = gr.Slider(
                        minimum=0,
                        maximum=50,
                        step=1,
                        elem_id="reranker_similarity_top_k",
                        label="重排序文本 Top-K (0 到 50)",
                    )

            with gr.Row(variant="panel"):
                with gr.Column():
                    _ = gr.Markdown(value="### **调试完成后保存将检索参数应用到线上**")
                    online_retrieval_config = gr.Code(
                        label="当前线上检索配置",
                        elem_id="online_retrieval_config",
                        language="json",
                        interactive=False,
                    )
                    save_retrieval_button = gr.Button(
                        value="保存当前参数并应用到线上",
                        elem_id="save_retrieval_button",
                        variant="primary",
                    )
                    _ = gr.Textbox(label="Save Info: ", container=False, visible=False)

            def change_weight(change_weight):
                return round(float(1 - change_weight), 2)

            def change_reranker_type(reranker_type):
                if reranker_type == "无重排序":
                    return {
                        model_reranker_col: gr.update(visible=False),
                    }
                elif reranker_type == "基于模型的重排序":
                    return {
                        model_reranker_col: gr.update(visible=True),
                    }
                else:
                    return {
                        model_reranker_col: gr.update(visible=False),
                    }

            reranker_type.input(
                fn=change_reranker_type,
                inputs=reranker_type,
                outputs=[model_reranker_col],
            )

            db_retrieval_elements = [
                retrieval_mode,
                reranker_type,
                similarity_top_k,
                similarity_threshold,
                reranker_similarity_threshold,
                reranker_model,
                reranker_similarity_top_k,
                online_retrieval_config,
                save_retrieval_button,
            ]
            components.extend(db_retrieval_elements)

        with gr.Column(scale=7):
            retrieval_test_chat_index = gr.Dropdown(
                choices=[],
                value="",
                label="\N{bookmark} 知识库名称",
                elem_id="retrieval_test_chat_index",
                allow_custom_value=True,
            )
            chatbot = gr.Chatbot(
                height=500, elem_id="retrieval_test_chatbot", type="messages"
            )
            with gr.Row():
                question = gr.Textbox(
                    label="在这里输入您的问题", elem_id="retrieval_test_question", scale=9
                )
            with gr.Row():
                submitBtn = gr.Button("提交", variant="primary")
                clearBtn = gr.Button("清空历史", variant="secondary")

            retrieval_chat_elements = {
                retrieval_mode,
                reranker_type,
                similarity_top_k,
                # image_similarity_top_k,
                similarity_threshold,
                reranker_similarity_threshold,
                reranker_model,
                reranker_similarity_top_k,
                save_retrieval_button,
                retrieval_test_chat_index,
                chatbot,
                question,
            }

            components.extend([retrieval_test_chat_index, chatbot, question])

            retrieval_test_chat_index.change(
                fn=show_retrieval_config,
                inputs=[retrieval_test_chat_index],
                outputs=[
                    online_retrieval_config,
                    retrieval_mode,
                    reranker_type,
                    similarity_top_k,
                    similarity_threshold,
                    reranker_similarity_threshold,
                    reranker_model,
                    reranker_similarity_top_k,
                ],
                api_name="show_retrieval_config",
            )

            retrieval_test_chat_index.change(
                fn=show_retrieval_config,
                inputs=[retrieval_test_chat_index],
                outputs=[
                    online_retrieval_config,
                    retrieval_mode,
                    reranker_type,
                    similarity_top_k,
                    similarity_threshold,
                    reranker_similarity_threshold,
                    reranker_model,
                    reranker_similarity_top_k,
                ],
                api_name="show_retrieval_config",
            )

            save_retrieval_button.click(
                fn=save_retrieval_config,
                inputs=retrieval_chat_elements,
                outputs=[online_retrieval_config],
                api_name="save_retrieval_button",
            )

            submitBtn.click(
                retrieval_test_respond,
                retrieval_chat_elements,
                [chatbot],
                api_name="retrieval_test_respond_clk",
            )
            question.submit(
                retrieval_test_respond,
                retrieval_chat_elements,
                [chatbot],
                api_name="retrieval_test_respond_q",
            )
            submitBtn.click(
                reset_textbox,
                [],
                [question],
                api_name="retrieval_test_reset_clk",
            )
            question.submit(
                reset_textbox,
                [],
                [question],
                api_name="retrieval_test_reset_q",
            )
            cur_tokens = gr.Textbox(label="\N{fire} 当前Tokens总数", visible=False)
            clearBtn.click(clear_history, [chatbot], [chatbot, cur_tokens])
        return components_to_dict(components)


def create_knowledgebase_qa_prompt_tab():
    components = []
    with gr.Row():
        with gr.Column():
            with gr.Row():
                knowledgebase_qa_prompt_index = gr.Dropdown(
                    choices=[],
                    value="",
                    label="\N{bookmark} 知识库名称",
                    elem_id="knowledgebase_qa_prompt_index",
                    allow_custom_value=True,
                )
            with gr.Row():
                knowledgebase_qa_system_prompt_template = gr.Textbox(
                    label="系统角色设定",
                    value="",
                    elem_id="knowledgebase_qa_system_prompt_template",
                    lines=4,
                    interactive=True,
                )
                knowledgebase_qa_task_prompt_template = gr.Textbox(
                    label="任务描述",
                    value="",
                    elem_id="knowledgebase_qa_task_prompt_template",
                    lines=10,
                    interactive=True,
                )
    with gr.Row():
        with gr.Column():
            save_knowledgebase_qa_prompt_btn = gr.Button(
                value="检查并保存配置",
                elem_id="save_knowledgebase_qa_prompt_btn",
                variant="primary",
            )
            save_knowledgebase_qa_prompt_state = gr.Textbox(
                label="Save Info: ", container=False, visible=True
            )
    components.extend(
        [
            knowledgebase_qa_prompt_index,
            knowledgebase_qa_system_prompt_template,
            knowledgebase_qa_task_prompt_template,
        ]
    )

    save_knowledgebase_qa_prompt_btn.click(
        fn=save_knowledgebase_qa_prompt_func,
        inputs=set(components),
        outputs=[save_knowledgebase_qa_prompt_state],
        api_name="save_knowledgebase_qa_prompt",
    )
    knowledgebase_qa_prompt_index.input(
        fn=show_knowledgebase_qa_prompt_templates,
        inputs=[knowledgebase_qa_prompt_index],
        outputs=[
            knowledgebase_qa_system_prompt_template,
            knowledgebase_qa_task_prompt_template,
        ],
        api_name="show_knowledgebase_qa_prompt_templates",
    )
    knowledgebase_qa_prompt_index.change(
        fn=show_knowledgebase_qa_prompt_templates,
        inputs=[knowledgebase_qa_prompt_index],
        outputs=[
            knowledgebase_qa_system_prompt_template,
            knowledgebase_qa_task_prompt_template,
        ],
        api_name="show_knowledgebase_qa_prompt_templates",
    )
    return components_to_dict(components)


def create_knowledgebase_tab() -> Dict[str, Any]:
    with gr.Tab("知识库设置"):
        knowledgebase_settings_elements = create_knowledgebase_settings_tab()
    with gr.Tab("文件管理"):
        with gr.Blocks():
            html = '<iframe src="./filebrowser" width="100%" height="1000" title="FileBrowser"></iframe>'
            gr.HTML(html)
    with gr.Tab("上传历史"):
        history_elements = create_upload_history()
    with gr.Tab("检索测试"):
        retrieval_test_elements = create_retrieval_test_tab()
    with gr.Tab("知识库问答提示词模板配置"):
        knowledgebase_qa_elements = create_knowledgebase_qa_prompt_tab()
    return {
        **knowledgebase_settings_elements,
        **history_elements,
        **retrieval_test_elements,
        **knowledgebase_qa_elements,
    }
