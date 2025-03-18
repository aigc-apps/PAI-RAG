from typing import Dict, Any, List
import gradio as gr
from pai_rag.app.web.rag_local_client import RagApiError, rag_client
from loguru import logger


def clear_history(chatbot):
    chatbot = []
    return chatbot, 0


def reset_textbox():
    return gr.update(value="")


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


async def respond(input_elements: List[Any]):
    update_dict = {}
    for element, value in input_elements.items():
        update_dict[element.elem_id] = value

    # empty input.
    if not update_dict["question"]:
        yield update_dict["chatbot"]
        return

    try:
        rag_client.patch_config(update_dict)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    chatbot = update_dict["chatbot"]
    query_type = update_dict["query_type"]
    question = update_dict["question"]
    q_msg = {"content": question, "role": "user"}
    chatbot.append(q_msg)
    is_streaming = update_dict["is_streaming"]
    index_name = update_dict["chat_index"]
    chat_model_id = update_dict["chat_model_id"]
    query_rewrite_model_id = update_dict["query_rewrite_model_id"]
    citation = update_dict["citation"]
    return_reference = update_dict["return_reference"]

    if chatbot is not None:
        chatbot.append(
            {"content": "", "role": "assistant", "metadata": {"status": "pending"}}
        )
        yield chatbot

    try:
        if query_type == "对话 (大模型)":
            response_gen = rag_client.query_llm(
                chat_messages=chatbot[:-1],
                stream=is_streaming,
                chat_model_id=chat_model_id,
                query_rewrite_model_id=query_rewrite_model_id,
            )
        elif query_type == "检索测试":
            response_gen = rag_client.query_vector(
                chatbot[:-1], question, index_name=index_name
            )

        elif query_type == "对话 (网络搜索)":
            response_gen = rag_client.query(
                chat_messages=chatbot[:-1],
                stream=is_streaming,
                citation=citation,
                search_web=True,
                return_reference=return_reference,
                chat_model_id=chat_model_id,
                query_rewrite_model_id=query_rewrite_model_id,
            )
        else:
            response_gen = rag_client.query(
                chat_messages=chatbot[:-1],
                stream=is_streaming,
                citation=citation,
                index_name=index_name,
                return_reference=return_reference,
                chat_model_id=chat_model_id,
                query_rewrite_model_id=query_rewrite_model_id,
            )

        is_thinking = False
        async for resp in response_gen:
            if resp.delta == "<think>":
                chatbot[-1]["metadata"]["title"] = "thinking..."
                chatbot[-1]["metadata"]["log"] = ""
                is_thinking = True

            elif resp.delta == "</think>":
                chatbot[-1]["metadata"]["title"] = "thought"
                chatbot[-1]["metadata"]["status"] = "done"
                is_thinking = False
                chatbot.append(
                    {
                        "content": "",
                        "role": "assistant",
                        "metadata": {"status": "pending"},
                    }
                )

            else:
                if is_thinking:
                    chatbot[-1]["metadata"]["log"] += resp.delta
                else:
                    chatbot[-1]["content"] += resp.delta
            yield chatbot

    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")
    except Exception as e:
        raise gr.Error(f"Error: {e}")
    finally:
        logger.info(f"Chatbot finished: {chatbot}")
        yield chatbot


def create_chat_tab() -> Dict[str, Any]:
    rag_config = rag_client.get_config()
    model_choices = [
        llm.model_id if llm.model_id else llm.model for llm in rag_config.llms
    ]
    model_name = rag_config.chat.model_id
    if not model_name:
        if len(model_choices) == 0:
            model_name = ""
        else:
            model_name = model_choices[0]
    with gr.Row():
        with gr.Column(scale=2):
            chat_model_id = gr.Dropdown(
                choices=model_choices,
                value=model_name,
                label="\N{bookmark} 对话模型ID",
                elem_id="chat_model_id",
            )
            chat_index = gr.Dropdown(
                choices=[],
                value="",
                label="\N{bookmark} 知识库名称",
                elem_id="chat_index",
                allow_custom_value=True,
            )
            query_type = gr.Radio(
                ["检索测试", "对话 (大模型)", "对话 (网络搜索)", "对话 (知识库)"],
                label="\N{fire} 对话方式",
                elem_id="query_type",
                value="对话 (知识库)",
            )
            is_streaming = gr.Checkbox(
                label="流式输出",
                info="开启流式输出",
                elem_id="is_streaming",
                value=True,
            )
            citation = gr.Checkbox(
                label="返回引用",
                info="在回答中返回引用编号",
                elem_id="citation",
                value=False,
            )
            need_image = gr.Checkbox(
                label="多模态推理",
                info="使用多模态大模型推理.",
                elem_id="need_image",
                visible=False,
            )
            return_reference = gr.Checkbox(
                label="展示参考资料",
                info="展示 RAG 和网页搜索的参考资料",
                elem_id="return_reference",
                visible=True,
                value=False,
            )
            default_web_search = gr.Checkbox(
                label="默认网络搜索",
                info="使用OpenAI调用时默认开启网络搜索",
                elem_id="default_web_search",
                value=False,
            )

            quer_rewrite_model_name = rag_config.query_rewrite.model_id
            if not quer_rewrite_model_name:
                if len(model_choices) == 0:
                    quer_rewrite_model_name = ""
                else:
                    quer_rewrite_model_name = model_choices[0]

            with gr.Column(visible=True) as qt_col:
                query_transform_argument = gr.Accordion("查询改写配置", open=False)
                with query_transform_argument:
                    enable_query_transform = gr.Checkbox(
                        label="开启查询改写",
                        elem_id="enable_query_transform",
                        container=True,
                    )
                    with gr.Row(
                        visible=False, elem_id="enable_query_transform_col"
                    ) as enable_query_transform_col:
                        query_rewrite_model_id = gr.Dropdown(
                            choices=model_choices,
                            value=quer_rewrite_model_name,
                            label="\N{bookmark} 查询改写模型ID",
                            elem_id="query_rewrite_model_id",
                        )
                        query_transform_template = gr.Textbox(
                            label="查询改写模板",
                            value="",
                            elem_id="query_transform_template",
                            lines=10,
                            interactive=True,
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

            with gr.Column(visible=True) as vs_col:
                vec_model_argument = gr.Accordion("向量检索参数设置", open=False)
                with vec_model_argument:
                    retrieval_mode = gr.Radio(
                        ["向量检索", "关键字检索", "混合检索"],
                        label="检索模式",
                        elem_id="retrieval_mode",
                    )

                    vector_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.7,
                        elem_id="vector_weight",
                        label="向量检索权重",
                        visible=(retrieval_mode == "混合检索"),
                    )
                    keyword_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=float(1 - vector_weight.value),
                        elem_id="keyword_weight",
                        label="关键字检索权重",
                        interactive=False,
                        visible=(retrieval_mode == "混合检索"),
                    )

                    similarity_top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        elem_id="similarity_top_k",
                        label="返回Top-K条文本结果 (0 到 100)",
                    )
                    image_similarity_top_k = gr.Slider(
                        minimum=0,
                        maximum=10,
                        step=1,
                        elem_id="image_similarity_top_k",
                        label="返回Top-K条图片结果 (0 到 10)",
                    )
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

                    def change_weight(change_weight):
                        return round(float(1 - change_weight), 2)

                    vector_weight.input(
                        fn=change_weight,
                        inputs=vector_weight,
                        outputs=[keyword_weight],
                    )

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

                    def change_retrieval_mode(retrieval_mode):
                        if retrieval_mode == "混合检索":
                            return {
                                vector_weight: gr.update(visible=True),
                                keyword_weight: gr.update(visible=True),
                            }
                        else:
                            return {
                                vector_weight: gr.update(visible=False),
                                keyword_weight: gr.update(visible=False),
                            }

                    reranker_type.input(
                        fn=change_reranker_type,
                        inputs=reranker_type,
                        outputs=[model_reranker_col],
                    )

                    retrieval_mode.input(
                        fn=change_retrieval_mode,
                        inputs=retrieval_mode,
                        outputs=[vector_weight, keyword_weight],
                    )

                vec_args = {
                    retrieval_mode,
                    reranker_type,
                    vector_weight,
                    keyword_weight,
                    similarity_top_k,
                    image_similarity_top_k,
                    similarity_threshold,
                    reranker_similarity_threshold,
                    reranker_model,
                    reranker_similarity_top_k,
                }

            with gr.Column(visible=True) as lc_col:
                prompt_argument = gr.Accordion("提示词模板", open=False)
                with prompt_argument:
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
                    # with gr.Tab(
                    #     "MultiModal Prompt", interactive=True
                    # ) as multimodal_prompt_col:
                    #     multimodal_qa_template = gr.Textbox(
                    #         label="Multi-modal Prompt Template",
                    #         value="",
                    #         elem_id="multimodal_qa_template",
                    #         lines=12,
                    #         interactive=True,
                    #     )
                    #     citation_multimodal_qa_template = gr.Textbox(
                    #         label="Citation Multi-modal Prompt Template",
                    #         value="",
                    #         elem_id="citation_multimodal_qa_template",
                    #         lines=12,
                    #         interactive=True,
                    #     )

            with gr.Column(visible=True) as llm_col:
                model_argument = gr.Accordion("LLM推理参数设置", open=False)
                with model_argument:
                    llm_temperature = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.001,
                        value=0.1,
                        elem_id="llm_temperature",
                        label="温度 (0 到 1)",
                    )
                llm_args = {llm_temperature}

            with gr.Column(visible=False) as search_col:
                search_model_argument = gr.Accordion("网络搜索参数设置", open=False)
                with search_model_argument:
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
                search_args = {
                    search_type,
                    search_api_key,
                    search_count,
                    search_lang,
                    aliyun_endpoint,
                    aliyun_access_key_id,
                    aliyun_access_key_secret,
                    serpapi_key,
                }
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

            cur_tokens = gr.Textbox(label="\N{fire} 当前Tokens总数", visible=False)

            def change_query_radio(query_type):
                if query_type == "检索测试":
                    return {
                        vs_col: gr.update(visible=True),
                        vec_model_argument: gr.update(open=True),
                        qt_col: gr.update(visible=True),
                        query_transform_argument: gr.update(open=False),
                        search_model_argument: gr.update(open=False),
                        search_col: gr.update(visible=False),
                        llm_col: gr.update(visible=False),
                        model_argument: gr.update(open=False),
                        lc_col: gr.update(visible=False),
                        prompt_argument: gr.update(open=False),
                        return_reference: gr.update(visible=False),
                    }
                elif query_type == "对话 (大模型)":
                    return {
                        vs_col: gr.update(visible=False),
                        vec_model_argument: gr.update(open=False),
                        qt_col: gr.update(visible=True),
                        query_transform_argument: gr.update(open=False),
                        search_model_argument: gr.update(open=False),
                        search_col: gr.update(visible=False),
                        llm_col: gr.update(visible=True),
                        model_argument: gr.update(open=True),
                        lc_col: gr.update(visible=True),
                        prompt_argument: gr.update(open=True),
                        return_reference: gr.update(visible=False),
                    }
                elif query_type == "对话 (知识库)":
                    return {
                        vs_col: gr.update(visible=True),
                        vec_model_argument: gr.update(open=False),
                        qt_col: gr.update(visible=True),
                        query_transform_argument: gr.update(open=False),
                        search_model_argument: gr.update(open=False),
                        search_col: gr.update(visible=False),
                        llm_col: gr.update(visible=True),
                        model_argument: gr.update(open=False),
                        lc_col: gr.update(visible=True),
                        prompt_argument: gr.update(open=True),
                        return_reference: gr.update(visible=True),
                    }
                elif query_type == "对话 (网络搜索)":
                    return {
                        vs_col: gr.update(visible=False),
                        vec_model_argument: gr.update(open=False),
                        qt_col: gr.update(visible=True),
                        query_transform_argument: gr.update(open=False),
                        search_model_argument: gr.update(open=True),
                        search_col: gr.update(visible=True),
                        prompt_argument: gr.update(open=True),
                        llm_col: gr.update(visible=False),
                        model_argument: gr.update(open=False),
                        lc_col: gr.update(visible=True),
                        return_reference: gr.update(visible=True),
                    }

            query_type.input(
                fn=change_query_radio,
                inputs=query_type,
                outputs=[
                    prompt_argument,
                    vs_col,
                    vec_model_argument,
                    qt_col,
                    query_transform_argument,
                    search_model_argument,
                    search_col,
                    llm_col,
                    model_argument,
                    lc_col,
                    return_reference,
                ],
            )

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(height=500, elem_id="chatbot", type="messages")
            with gr.Row():
                question = gr.Textbox(label="在这里输入您的问题", elem_id="question", scale=9)
            with gr.Row():
                submitBtn = gr.Button("提交", variant="primary")
                clearBtn = gr.Button("清空历史", variant="secondary")

        chat_args = (
            {
                chat_model_id,
                default_web_search,
                enable_query_transform,
                query_rewrite_model_id,
                query_transform_template,
                system_role_template,
                custom_prompt_template,
                question,
                query_type,
                chatbot,
                is_streaming,
                citation,
                need_image,
                chat_index,
                return_reference,
            }
            .union(vec_args)
            .union(llm_args)
            .union(search_args)
        )

        submitBtn.click(
            respond,
            chat_args,
            [chatbot],
            api_name="respond_clk",
        )
        question.submit(
            respond,
            chat_args,
            [chatbot],
            api_name="respond_q",
        )
        submitBtn.click(
            reset_textbox,
            [],
            [question],
            api_name="reset_clk",
        )
        question.submit(
            reset_textbox,
            [],
            [question],
            api_name="reset_q",
        )

        clearBtn.click(clear_history, [chatbot], [chatbot, cur_tokens])
        return {
            default_web_search.elem_id: default_web_search,
            chat_index.elem_id: chat_index,
            similarity_top_k.elem_id: similarity_top_k,
            image_similarity_top_k.elem_id: image_similarity_top_k,
            need_image.elem_id: need_image,
            retrieval_mode.elem_id: retrieval_mode,
            reranker_type.elem_id: reranker_type,
            reranker_model.elem_id: reranker_model,
            vector_weight.elem_id: vector_weight,
            keyword_weight.elem_id: keyword_weight,
            similarity_threshold.elem_id: similarity_threshold,
            reranker_similarity_threshold.elem_id: reranker_similarity_threshold,
            reranker_similarity_top_k.elem_id: reranker_similarity_top_k,
            enable_query_transform.elem_id: enable_query_transform,
            query_transform_template.elem_id: query_transform_template,
            chat_model_id.elem_id: chat_model_id,
            query_rewrite_model_id.elem_id: query_rewrite_model_id,
            system_role_template.elem_id: system_role_template,
            custom_prompt_template.elem_id: custom_prompt_template,
            search_lang.elem_id: search_lang,
            search_api_key.elem_id: search_api_key,
            serpapi_key.elem_id: serpapi_key,
            search_count.elem_id: search_count,
            search_type.elem_id: search_type,
            aliyun_endpoint.elem_id: aliyun_endpoint,
            aliyun_access_key_id.elem_id: aliyun_access_key_id,
            aliyun_access_key_secret.elem_id: aliyun_access_key_secret,
            model_reranker_col.elem_id: model_reranker_col,
            llm_temperature.elem_id: llm_temperature,
            query_type.elem_id: query_type,
        }
