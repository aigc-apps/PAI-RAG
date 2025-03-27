from typing import Dict, Any
import gradio as gr
from pai_rag.app.web.ui_constants import EMBEDDING_API_KEY_DICT
from pai_rag.app.web.utils import components_to_dict
from pai_rag.app.web.index_utils import index_related_component_keys
from pai_rag.app.web.tabs.vector_db_panel import create_vector_db_panel
import pai_rag.app.web.event_listeners as ev_listeners
from pai_rag.app.web.rag_local_client import rag_client


def create_setting_tab() -> Dict[str, Any]:
    components = []
    with gr.Row():
        with gr.Column(variant="panel"):
            with gr.Column(scale=5):
                _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **知识库**")

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
            vector_db_elems, vector_db_components = create_vector_db_panel()

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

        ############################ llms settings start ############################
        rag_config = rag_client.get_config()
        with gr.Column(variant="panel"):
            # 模型选择区域
            model_choices = [
                llm.model_id if llm.model_id else llm.model for llm in rag_config.llms
            ] + ["NEW"]

            with gr.Row():
                llm_model = gr.Dropdown(
                    label="模型配置",
                    choices=model_choices,
                    value="NEW"
                    if not rag_config.llms and len(rag_config.llms) == 0
                    else model_choices[0],
                    interactive=True,
                    elem_id="llm_model",
                    allow_custom_value=False,
                )

                delete_btn = gr.Button(
                    "删除模型", visible=bool(rag_config.llms), variant="secondary"
                )

            # 新增/编辑配置区域
            with gr.Row(visible=True) as config_row:
                with gr.Column():
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            value=rag_config.llms[0].base_url
                            if rag_config.llms
                            else "",
                            label="URL",
                            placeholder="Open AI compatible url, e.g. https://api.openai.com/v1",
                            interactive=True,
                            scale=1,
                        )
                        llm_api_key = gr.Textbox(
                            value=rag_config.llms[0].api_key if rag_config.llms else "",
                            label="密钥",
                            type="password",
                            interactive=True,
                            scale=1,
                        )

                    with gr.Row():
                        llm_model_name = gr.Textbox(
                            value=rag_config.llms[0].model if rag_config.llms else "",
                            label="模型名称",
                            placeholder="模型名称, e.g. qwen-max",
                            interactive=True,
                            scale=1,
                        )
                        llm_model_id = gr.Textbox(
                            value=rag_config.llms[0].model_id
                            if rag_config.llms
                            else "",
                            label="模型ID",
                            placeholder="模型ID(建议模型ID与模型名称保持一致，或填写一个您偏好的、便于区分的名称), e.g. model_1",
                            interactive=True,
                            scale=1,
                        )

                    # 第三行：多模态支持
                    with gr.Row():
                        with gr.Column():
                            llm_vision_support = gr.Checkbox(
                                value=rag_config.llms[0].vision_support
                                if rag_config.llms
                                else False,
                                label="是否支持多模态",
                                elem_id="vision_support",
                                container=True,  # 让复选框有背景容器
                                scale=1,
                            )
                            is_reasoning_models = gr.Checkbox(
                                value=rag_config.llms[0].is_reasoning_models
                                if rag_config.llms
                                else False,
                                label="是否为推理模型(Reasoning Models)",
                                elem_id="is_reasoning_models",
                                container=True,  # 让复选框有背景容器
                                scale=1,
                            )
            save_btn = gr.Button("保存模型配置", variant="primary")

            llm_model.change(
                fn=ev_listeners.update_llms,
                inputs=llm_model,
                outputs=[
                    config_row,
                    delete_btn,
                    llm_base_url,
                    llm_api_key,
                    llm_model_name,
                    llm_model_id,
                    llm_vision_support,
                    is_reasoning_models,
                ],
            )

            save_btn.click(
                fn=ev_listeners.save_new_llm,
                inputs=[
                    llm_model,
                    llm_model_name,
                    llm_base_url,
                    llm_api_key,
                    llm_model_id,
                    llm_vision_support,
                    is_reasoning_models,
                ],
                outputs=[llm_model, delete_btn],
            )

            delete_btn.click(
                fn=ev_listeners.delete_llm,
                inputs=llm_model,
                outputs=[llm_model, config_row, delete_btn],
            )
            ############################ llms settings end  ############################

            with gr.Column(scale=5, variant="panel"):
                _ = gr.Markdown(value="\N{WHITE MEDIUM STAR} **(可选) 图片OSS存储配置**")
                use_oss = gr.Checkbox(
                    label="使用图片OSS存储",
                    elem_id="use_oss",
                    container=False,
                )
                with gr.Row(visible=False, elem_id="use_oss_col") as use_oss_col:
                    oss_bucket = gr.Textbox(
                        label="OSS存储空间",
                        elem_id="oss_bucket",
                    )
                    oss_endpoint = gr.Textbox(
                        label="OSS访问域名",
                        elem_id="oss_endpoint",
                        placeholder="oss-cn-hangzhou.aliyuncs.com",
                    )
                    oss_ak = gr.Textbox(
                        label="AccessKey ID",
                        elem_id="oss_ak",
                        type="password",
                    )
                    oss_sk = gr.Textbox(
                        label="AccessKey Secret",
                        elem_id="oss_sk",
                        type="password",
                    )
                use_oss.input(
                    fn=ev_listeners.change_use_oss,
                    inputs=use_oss,
                    outputs=use_oss_col,
                )

            with gr.Column(scale=5, variant="panel"):
                _ = gr.Markdown(
                    value="\N{WHITE MEDIUM STAR} **(可选) LLM安全护栏 [doc](https://help.aliyun.com/document_detail/464388.html?spm=a2c4g.11186623.help-menu-28415.d_1_0.18923104V0TR1X)**"
                )
                enable_guardrail = gr.Checkbox(
                    label="开启LLM安全护栏",
                    elem_id="enable_guardrail",
                    container=False,
                )
                with gr.Row(visible=False, elem_id="guardrail_col") as guardrail_col:
                    guardrail_endpoint = gr.Textbox(
                        label="接入地址",
                        elem_id="guardrail_endpoint",
                        placeholder="green-cip.cn-hangzhou.aliyuncs.com",
                    )
                    guardrail_region = gr.Textbox(
                        label="地域",
                        elem_id="guardrail_region",
                        placeholder="cn-hangzhou",
                    )
                    guardrail_ak = gr.Textbox(
                        label="AccessKey ID",
                        elem_id="guardrail_ak",
                    )
                    guardrail_sk = gr.Textbox(
                        label="AccessKey Secret",
                        elem_id="guardrail_sk",
                        type="password",
                    )
                enable_guardrail.input(
                    fn=ev_listeners.change_enable_guardrail,
                    inputs=enable_guardrail,
                    outputs=guardrail_col,
                )

            oss_components = [
                use_oss,
                oss_ak,
                oss_sk,
                oss_endpoint,
                oss_bucket,
                guardrail_ak,
                guardrail_sk,
                guardrail_region,
                guardrail_endpoint,
            ]

            components.extend(oss_components)
            components.append(llm_model)

            # use_mllm.input(
            #     fn=ev_listeners.choose_use_mllm,
            #     inputs=use_mllm,
            #     outputs=[use_mllm_col],
            # )

            save_oss_btn = gr.Button("保存OSS配置", variant="primary")
            save_state = gr.Textbox(
                label="Connection Info: ", container=False, visible=False
            )
            save_oss_btn.click(
                fn=ev_listeners.save_config,
                inputs=set(oss_components),
                outputs=[oss_ak, oss_sk, save_state],
                api_name="save_config",
            )
    elems = components_to_dict(components)
    elems.update(vector_db_components)
    elems.update({use_oss_col.elem_id: use_oss_col})
    return elems
