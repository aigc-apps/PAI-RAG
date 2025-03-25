import gradio as gr
import re
from pai_rag.app.web.utils import components_to_dict
from pai_rag.app.web.rag_local_client import RagApiError, rag_client
from typing import Any, List
import datetime


def check_variables_in_string(text, variables):
    missing_variables = [var for var in variables if f"{{{var}}}" not in text]
    if missing_variables:
        raise ValueError(f"以下变量名缺失: {', '.join(missing_variables)}")


def is_valid_comma_separated_string(s):
    # 使用正则表达式校验
    # ^[\u4e00-\u9fa5]+(,[\u4e00-\u9fa5]+)*$ 的含义：
    # ^ 开头
    # [\u4e00-\u9fa5]+ 至少一个中文字符
    # (,[\u4e00-\u9fa5]+)* 零个或多个由逗号分隔的中文字符
    # $ 结尾
    pattern = r"^[\u4e00-\u9fa5]+(,[\u4e00-\u9fa5]+)*$"
    return bool(re.match(pattern, s))


list_news_pmt_required_variables = ["topics_str", "news_list_str"]
chat_news_pmt_required_variables = ["content", "prompt", "answerLength"]


def save_news_extension_config(input_elements: List[Any]):
    try:
        update_dict = {}
        for element, value in input_elements.items():
            if element.elem_id == "list_news_pmt":
                try:
                    check_variables_in_string(value, list_news_pmt_required_variables)
                except RagApiError:
                    return gr.Error("查询全局热门新闻的提示词模板保存出错，缺少变量")
            elif element.elem_id == "chat_news_pmt":
                try:
                    check_variables_in_string(value, chat_news_pmt_required_variables)
                except RagApiError:
                    return gr.Error("查询某个具体新闻的提示词模板保存出错，缺少变量")
            elif element.elem_id == "domain_list":
                if not is_valid_comma_separated_string(value):
                    raise ValueError(f"{value} is invalid domain list.")
            update_dict[element.elem_id] = value
        rag_client.patch_config(update_dict)

        return gr.update(
            value=f"[{datetime.datetime.now()}] News configuration saved successfully!",
            visible=True,
        )
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def create_news_extension_tab():
    rag_config = rag_client.get_config()
    model_choices = [
        llm.model_id if llm.model_id else llm.model for llm in rag_config.llms
    ]
    news_extension_model_name = rag_config.news_extension.model_id
    if not news_extension_model_name:
        if len(model_choices) == 0:
            news_extension_model_name = ""
        else:
            news_extension_model_name = model_choices[0]

    with gr.Row():
        with gr.Column(scale=2):
            _ = gr.Markdown(
                value="## \N{WHITE MEDIUM STAR} **配置阿里云百炼-[全妙Agent](https://bailian.console.aliyun.com/?spm=5176.29619931.J__Z58Z6CX7MY__Ll8p1ZOR.1.2a2e59fcymfNz1#/app/app-market/quanmiao/news-broadcast)**"
            )
            with gr.Column():
                news_extension_model_id = gr.Dropdown(
                    choices=model_choices,
                    value=news_extension_model_name,
                    label="\N{bookmark} 新闻Agent模型ID",
                    elem_id="news_extension_model_id",
                    interactive=True,
                )
                bailian_workspaceid = gr.Textbox(
                    label="阿里云百炼工作空间ID",
                    elem_id="bailian_workspaceid",
                    interactive=True,
                )
                bailian_ak = gr.Textbox(
                    label="AccessKey ID",
                    elem_id="bailian_ak",
                    type="password",
                    interactive=True,
                )
                bailian_sk = gr.Textbox(
                    label="AccessKey Secret",
                    elem_id="bailian_sk",
                    type="password",
                    interactive=True,
                )
        with gr.Column(scale=8):
            _ = gr.Markdown(value="## \N{WHITE MEDIUM STAR} **调整提示词模板**")
            with gr.Row(elem_id="news_prompts"):
                with gr.Column():
                    _ = gr.Markdown(value="### **查询全局热门新闻**")
                    top_news_count = gr.Number(
                        label="全部新闻数量限制",
                        value=10,
                        elem_id="top_news_count",
                        interactive=True,
                    )
                    domain_list = gr.Textbox(
                        label="新闻领域集合(用','分隔，如: 科技,娱乐)",
                        placeholder="新闻领域集合，用','分隔，如: 科技,娱乐",
                        elem_id="domain_list",
                        interactive=True,
                    )
                    list_news_pmt = gr.Textbox(
                        label="查询全局热门新闻的提示词模板",
                        value="",
                        elem_id="list_news_pmt",
                        lines=10,
                        interactive=True,
                    )
                with gr.Column():
                    _ = gr.Markdown(value="### **查询某个具体新闻**")
                    # chat_news_answer_len = gr.Number(
                    #     label="答案的返回长度限制",
                    #     value=200,
                    #     elem_id="chat_news_answer_len",
                    #     interactive=True,
                    # )
                    chat_news_pmt = gr.Textbox(
                        label="查询某个具体新闻的提示词模板",
                        value="",
                        elem_id="chat_news_pmt",
                        lines=15,
                        interactive=True,
                    )

    with gr.Row():
        with gr.Column():
            save_news_extension_btn = gr.Button(
                value="检查并保存配置",
                elem_id="save_news_extension_config_btn",
                variant="primary",
            )
            save_state = gr.Textbox(label="Save Info: ", container=False, visible=True)

    components = [
        news_extension_model_id,
        bailian_workspaceid,
        bailian_ak,
        bailian_sk,
        top_news_count,
        domain_list,
        list_news_pmt,
        # chat_news_answer_len,
        chat_news_pmt,
        save_news_extension_btn,
    ]
    save_news_extension_btn.click(
        fn=save_news_extension_config,
        inputs=set(components),
        outputs=[save_state],
        api_name="save_news_extension_btn",
    )
    elems = components_to_dict(components)
    return elems
