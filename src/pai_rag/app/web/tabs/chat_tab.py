from typing import Dict, Any, List
import gradio as gr
from pai_rag.app.web.rag_local_client import RagApiError, rag_client
from loguru import logger


def clear_history(chatbot):
    chatbot = []
    return chatbot, 0


def reset_textbox():
    return gr.update(value="")


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
    query_types = update_dict["query_types"]
    question = update_dict["question"]
    q_msg = {"content": question, "role": "user"}
    chatbot.append(q_msg)
    is_streaming = update_dict["is_streaming"]
    index_name = update_dict["chat_index"]
    chat_model_id = update_dict["chat_model_id"]
    citation = update_dict["citation"]
    return_reference = update_dict["return_reference"]
    temperature = update_dict["llm_temperature"]

    if chatbot is not None:
        chatbot.append(
            {"content": "", "role": "assistant", "metadata": {"status": "pending"}}
        )
        yield chatbot

    chat_knowledgebase = True if "查询知识库" in query_types else False
    search_web = True if "联网搜索" in query_types else False
    chat_llm = True if "大模型" in query_types else False
    chat_agent = True if "agent" in query_types else False
    chat_db = True if "查询数据库" in query_types else False
    chat_news = True if "新闻工具" in query_types else False

    try:
        response_gen = rag_client.query(
            chat_messages=chatbot[:-1],
            stream=is_streaming,
            citation=citation,
            index_name=index_name,
            return_reference=return_reference,
            chat_model_id=chat_model_id,
            temperature=temperature,
            chat_knowledgebase=chat_knowledgebase,
            search_web=search_web,
            chat_db=chat_db,
            chat_agent=chat_agent,
            chat_llm=chat_llm,
            chat_news=chat_news,
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
        with gr.Column(scale=1):
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
            # query_type = gr.Radio(
            #     ["对话 (大模型)", "对话 (网络搜索)", "对话 (知识库)"],
            #     label="\N{fire} 对话方式",
            #     elem_id="query_type",
            #     value="对话 (知识库)",
            # )
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
                visible=False,
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
            llm_temperature = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.001,
                value=0.1,
                elem_id="llm_temperature",
                label="LLM推理参数设置",
                info="温度 (0 到 1)",
            )

            quer_rewrite_model_name = rag_config.query_rewrite.model_id
            if not quer_rewrite_model_name:
                if len(model_choices) == 0:
                    quer_rewrite_model_name = ""
                else:
                    quer_rewrite_model_name = model_choices[0]

            # with gr.Column(visible=True) as llm_col:
            #     model_argument = gr.Accordion("LLM推理参数设置", open=True)
            #     with model_argument:
            #         llm_temperature = gr.Slider(
            #             minimum=0,
            #             maximum=1,
            #             step=0.001,
            #             value=0.1,
            #             elem_id="llm_temperature",
            #             label="温度 (0 到 1)",
            #         )
            llm_args = {llm_temperature}

            # with gr.Column(visible=False) as search_col:
            #     search_model_argument = gr.Accordion("网络搜索参数设置", open=False)
            #     with search_model_argument:
            #         search_type = gr.Radio(
            #             ["bing", "aliyun", "google"],
            #             label="搜索引擎",
            #             elem_id="search_type",
            #         )
            #         serpapi_key_tips = gr.Markdown(
            #             value="如何获取 [SerpAPI Key](https://serpapi.com)"
            #         )
            #         search_api_key = gr.Text(
            #             label="Bing API Key",
            #             value="",
            #             type="password",
            #             elem_id="search_api_key",
            #         )
            #         serpapi_key = gr.Text(
            #             label="SerpAPI Key",
            #             value="",
            #             type="password",
            #             elem_id="serpapi_key",
            #         )
            #         search_count = gr.Slider(
            #             label="搜索数量",
            #             minimum=5,
            #             maximum=50,
            #             step=1,
            #             elem_id="search_count",
            #         )
            #         search_lang = gr.Radio(
            #             label="语言",
            #             choices=["zh-CN", "en-US"],
            #             value="zh-CN",
            #             elem_id="search_lang",
            #         )
            #         aliyun_endpoint = gr.Text(
            #             label="Endpoint", value="", elem_id="aliyun_endpoint"
            #         )
            #         aliyun_access_key_id = gr.Text(
            #             label="AccessKey ID", value="", elem_id="aliyun_access_key_id"
            #         )
            #         aliyun_access_key_secret = gr.Text(
            #             label="AccessKey Secret",
            #             value="",
            #             type="password",
            #             elem_id="aliyun_access_key_secret",
            #         )
            #     search_args = {
            #         search_type,
            #         search_api_key,
            #         search_count,
            #         search_lang,
            #         aliyun_endpoint,
            #         aliyun_access_key_id,
            #         aliyun_access_key_secret,
            #         serpapi_key,
            #     }
            #     search_type.input(
            #         fn=change_search_model_argument,
            #         inputs=[search_type],
            #         outputs=[
            #             search_api_key,
            #             search_count,
            #             search_lang,
            #             aliyun_endpoint,
            #             aliyun_access_key_id,
            #             aliyun_access_key_secret,
            #             serpapi_key,
            #             serpapi_key_tips,
            #         ],
            #     )

            cur_tokens = gr.Textbox(label="\N{fire} 当前Tokens总数", visible=False)

            # def change_query_radio(query_type):
            #     if query_type == "检索测试":
            #         return {
            #             search_model_argument: gr.update(open=False),
            #             search_col: gr.update(visible=False),
            #             llm_col: gr.update(visible=False),
            #             model_argument: gr.update(open=False),
            #             return_reference: gr.update(visible=False),
            #         }
            #     elif query_type == "对话 (大模型)":
            #         return {
            #             search_model_argument: gr.update(open=False),
            #             search_col: gr.update(visible=False),
            #             llm_col: gr.update(visible=True),
            #             model_argument: gr.update(open=True),
            #             return_reference: gr.update(visible=False),
            #         }
            #     elif query_type == "对话 (知识库)":
            #         return {
            #             search_model_argument: gr.update(open=False),
            #             search_col: gr.update(visible=False),
            #             llm_col: gr.update(visible=True),
            #             model_argument: gr.update(open=False),
            #             return_reference: gr.update(visible=True),
            #         }
            #     elif query_type == "对话 (网络搜索)":
            #         return {
            #             search_model_argument: gr.update(open=True),
            #             search_col: gr.update(visible=True),
            #             llm_col: gr.update(visible=True),
            #             model_argument: gr.update(open=False),
            #             return_reference: gr.update(visible=True),
            #         }

            # query_type.input(
            #     fn=change_query_radio,
            #     inputs=query_type,
            #     outputs=[
            #         search_model_argument,
            #         search_col,
            #         llm_col,
            #         model_argument,
            #         return_reference,
            #     ],
            # )

        with gr.Column(scale=9):
            chatbot = gr.Chatbot(height=500, elem_id="chatbot", type="messages")
            with gr.Row():
                with gr.Column(variant="panel"):
                    query_types = gr.CheckboxGroup(
                        ["大模型", "联网搜索", "查询知识库", "查询数据库", "新闻工具"],
                        # ["大模型", "联网搜索", "查询知识库", "查询数据库", "agent", "新闻工具"],
                        label="使用更多工具",
                        elem_id="query_types",
                    )
                    question = gr.Textbox(
                        label="在这里输入您的问题", elem_id="question", scale=9
                    )
            with gr.Row():
                submitBtn = gr.Button("提交", variant="primary")
                clearBtn = gr.Button("清空历史", variant="secondary")

        chat_args = (
            {
                chat_model_id,
                question,
                query_types,
                chatbot,
                is_streaming,
                citation,
                need_image,
                chat_index,
                return_reference,
            }.union(llm_args)
            # .union(search_args)
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
            chat_index.elem_id: chat_index,
            need_image.elem_id: need_image,
            chat_model_id.elem_id: chat_model_id,
            llm_temperature.elem_id: llm_temperature,
            query_types.elem_id: query_types,
        }
