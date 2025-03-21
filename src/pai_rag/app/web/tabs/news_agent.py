import gradio as gr
from pai_rag.app.web.utils import components_to_dict


def create_news_agent_tab():
    with gr.Row():
        with gr.Column(scale=3):
            _ = gr.Markdown(
                value="## \N{WHITE MEDIUM STAR} **配置阿里云百炼-[全妙Agent](https://bailian.console.aliyun.com/?spm=5176.29619931.J__Z58Z6CX7MY__Ll8p1ZOR.1.2a2e59fcymfNz1#/app/app-market/quanmiao/news-broadcast)**"
            )
            with gr.Column(elem_id="use_bailian_col"):
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
        with gr.Column(scale=7):
            _ = gr.Markdown(value="## \N{WHITE MEDIUM STAR} **调整提示词模板**")
            with gr.Column(elem_id="news_prompts"):
                _ = gr.Markdown(value="### **查询全局热门新闻**")
                list_news_pmt = gr.Textbox(
                    label="查询全局热门新闻的提示词模板",
                    value="",
                    elem_id="list_news_pmt",
                    lines=10,
                    interactive=True,
                )
                _ = gr.Markdown(value="### **查询某个具体新闻**")
                answer_len = gr.Number(
                    label="答案的返回长度限制",
                    value=200,
                    elem_id="answer_len",
                    interactive=True,
                )
                chat_news_pmt = gr.Textbox(
                    label="查询某个具体新闻的提示词模板",
                    value="",
                    elem_id="chat_news_pmt",
                    lines=10,
                    interactive=True,
                )
    components = [
        bailian_workspaceid,
        bailian_ak,
        bailian_sk,
        list_news_pmt,
        answer_len,
        chat_news_pmt,
    ]
    elems = components_to_dict(components)
    return elems
