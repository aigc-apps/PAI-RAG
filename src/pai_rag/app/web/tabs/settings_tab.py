from typing import Dict, Any
import gradio as gr
from pai_rag.app.web.utils import components_to_dict
import pai_rag.app.web.event_listeners as ev_listeners
from pai_rag.app.web.rag_local_client import rag_client


def create_setting_tab() -> Dict[str, Any]:
    components = []
    with gr.Row():
        with gr.Tab("模型及存储配置"):
            rag_config = rag_client.get_config()
            with gr.Row():
                with gr.Column(variant="panel"):
                    # 模型选择区域
                    model_choices = [
                        llm.model_id if llm.model_id else llm.model
                        for llm in rag_config.llms
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
                                    value=rag_config.llms[0].api_key
                                    if rag_config.llms
                                    else "",
                                    label="密钥",
                                    type="password",
                                    interactive=True,
                                    scale=1,
                                )

                            with gr.Row():
                                llm_model_name = gr.Textbox(
                                    value=rag_config.llms[0].model
                                    if rag_config.llms
                                    else "",
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
                                llm_vision_support = gr.Checkbox(
                                    value=rag_config.llms[0].vision_support
                                    if rag_config.llms
                                    else False,
                                    label="是否支持多模态",
                                    elem_id="vision_support",
                                    container=True,  # 让复选框有背景容器
                                    scale=1,
                                )
                                llm_reasoning_support = gr.Checkbox(
                                    value=rag_config.llms[0].is_reasoning_model
                                    if rag_config.llms
                                    else False,
                                    label="是否支持深度思考",
                                    elem_id="reasoning_support",
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
                            llm_reasoning_support,
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
                            llm_reasoning_support,
                        ],
                        outputs=[llm_model, delete_btn],
                    )

                    delete_btn.click(
                        fn=ev_listeners.delete_llm,
                        inputs=llm_model,
                        outputs=[llm_model, config_row, delete_btn],
                    )
                    ############################ llms settings end  ############################

                with gr.Column():
                    with gr.Column(scale=5, variant="panel"):
                        _ = gr.Markdown(
                            value="\N{WHITE MEDIUM STAR} **(可选) 图片OSS存储配置**"
                        )
                        use_oss = gr.Checkbox(
                            label="使用图片OSS存储",
                            elem_id="use_oss",
                            container=False,
                        )
                        with gr.Row(
                            visible=False, elem_id="use_oss_col"
                        ) as use_oss_col:
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
                        with gr.Row(
                            visible=False, elem_id="guardrail_col"
                        ) as guardrail_col:
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
        with gr.Tab("大模型对话提示词模板配置"):
            with gr.Row():
                system_role_template = gr.Textbox(
                    label="系统角色设定",
                    value="",
                    elem_id="system_role_template",
                    lines=4,
                    interactive=True,
                )
                custom_prompt_template = gr.Textbox(
                    label="任务描述",
                    value="",
                    elem_id="custom_prompt_template",
                    lines=10,
                    interactive=True,
                )
            with gr.Row():
                with gr.Column():
                    save_prompt_btn = gr.Button(
                        value="检查并保存配置",
                        elem_id="save_prompt_btn",
                        variant="primary",
                    )
                    save_prompt_state = gr.Textbox(
                        label="Save Info: ", container=False, visible=True
                    )
            components.extend(
                [system_role_template, custom_prompt_template, save_prompt_btn]
            )
            pmt_components = [
                system_role_template,
                custom_prompt_template,
            ]
            save_prompt_btn.click(
                fn=ev_listeners.save_pmt_cfg_func,
                inputs=set(pmt_components),
                outputs=[save_prompt_state],
                api_name="save_prompt_config",
            )

        with gr.Tab("查询改写配置"):
            quer_rewrite_model_name = rag_config.query_rewrite.model_id
            if not quer_rewrite_model_name:
                if len(model_choices) == 0:
                    quer_rewrite_model_name = ""
                else:
                    quer_rewrite_model_name = model_choices[0]
            with gr.Column(visible=True):
                enable_query_transform = gr.Checkbox(
                    label="开启查询改写",
                    elem_id="enable_query_transform",
                    container=True,
                )
                with gr.Row(
                    visible=False, elem_id="enable_query_transform_col"
                ) as enable_query_transform_col:
                    with gr.Column():
                        _ = gr.Markdown("### 查询改写配置")
                        query_rewrite_model_id = gr.Dropdown(
                            choices=model_choices,
                            value=quer_rewrite_model_name,
                            label="\N{bookmark} 查询改写模型ID",
                            elem_id="query_rewrite_model_id",
                            interactive=True,
                        )
                        rewrite_base_prompt = gr.Textbox(
                            label="查询改写提示词模板",
                            value="",
                            elem_id="rewrite_base_prompt",
                            lines=10,
                            interactive=True,
                        )
                    with gr.Column():
                        _ = gr.Markdown("### 使用工具的提示词模板")
                        with gr.Tab(label="大模型对话"):
                            rewrite_llm_prompt = gr.Textbox(
                                label="大模型对话",
                                value="",
                                elem_id="rewrite_llm_prompt",
                                lines=10,
                                interactive=True,
                            )
                        with gr.Tab(label="知识库问答"):
                            rewrite_knowledgebase_prompt = gr.Textbox(
                                label="知识库问答",
                                value="",
                                elem_id="rewrite_knowledgebase_prompt",
                                lines=10,
                                interactive=True,
                            )
                        with gr.Tab(label="联网搜索"):
                            rewrite_search_prompt = gr.Textbox(
                                label="联网搜索",
                                value="",
                                elem_id="rewrite_search_prompt",
                                lines=10,
                                interactive=True,
                            )
                        with gr.Tab(label="工具调用", visible=False):
                            rewrite_agent_prompt = gr.Textbox(
                                label="工具调用",
                                value="",
                                elem_id="rewrite_agent_prompt",
                                lines=10,
                                interactive=True,
                            )
                        with gr.Tab(label="数据库查询"):
                            rewrite_db_prompt = gr.Textbox(
                                label="数据库查询",
                                value="",
                                elem_id="rewrite_db_prompt",
                                lines=10,
                                interactive=True,
                            )
                        with gr.Tab(label="新闻查询"):
                            rewrite_news_prompt = gr.Textbox(
                                label="新闻查询",
                                value="",
                                elem_id="rewrite_news_prompt",
                                lines=10,
                                interactive=True,
                            )
                        with gr.Tab(label="MCP工具调用"):
                            rewrite_mcp_prompt = gr.Textbox(
                                label="MCP工具调用",
                                value="",
                                elem_id="rewrite_mcp_prompt",
                                lines=10,
                                interactive=True,
                            )
                with gr.Row():
                    with gr.Column():
                        save_query_transform_btn = gr.Button(
                            value="检查并保存配置",
                            elem_id="save_query_transform_btn",
                            variant="primary",
                        )
                        save_query_transform_state = gr.Textbox(
                            label="Save Info: ", container=False, visible=True
                        )

                    query_transform_pmt_components = [
                        enable_query_transform,
                        query_rewrite_model_id,
                        rewrite_base_prompt,
                        rewrite_llm_prompt,
                        rewrite_knowledgebase_prompt,
                        rewrite_agent_prompt,
                        rewrite_search_prompt,
                        rewrite_db_prompt,
                        rewrite_mcp_prompt,
                        rewrite_news_prompt,
                        save_query_transform_btn,
                    ]
                    save_query_transform_btn.click(
                        fn=ev_listeners.save_query_transform_cfg,
                        inputs=set(query_transform_pmt_components),
                        outputs=[save_query_transform_state],
                        api_name="save_query_transform_cfg",
                    )

                def change_query_transform_parameter(enable_query_transform):
                    if enable_query_transform:
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)

                enable_query_transform.change(
                    fn=change_query_transform_parameter,
                    inputs=enable_query_transform,
                    outputs=enable_query_transform_col,
                )
            components.extend(
                [
                    enable_query_transform,
                    query_rewrite_model_id,
                    rewrite_base_prompt,
                    rewrite_llm_prompt,
                    rewrite_knowledgebase_prompt,
                    rewrite_agent_prompt,
                    rewrite_search_prompt,
                    rewrite_news_prompt,
                    rewrite_db_prompt,
                    rewrite_mcp_prompt,
                    save_query_transform_btn,
                ]
            )
    elems = components_to_dict(components)
    # elems.update(vector_db_components)
    elems.update({use_oss_col.elem_id: use_oss_col})
    return elems
