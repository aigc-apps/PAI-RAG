import os
from typing import Dict, Any
import gradio as gr
from pai_rag.app.web.rag_client import rag_client
import pandas as pd
import shutil


def upload_file_fn(input_file):
    if input_file is None:
        return None

    persist_path = "./localdata/data_analysis"
    if not os.path.exists(persist_path):
        os.mkdir(persist_path)
    else:
        # 清空目录中的文件
        for filename in os.listdir(persist_path):
            file_path = os.path.join(persist_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    # 指定持久化存储位置
    file_name = os.path.basename(input_file.name)
    destination_path = os.path.join(persist_path, file_name)
    # 复制文件
    shutil.copy(input_file.name, destination_path)

    update_dict = {
        # "retrieval_mode": "data_analysis",
        "data_analysis_file_path": destination_path,
    }
    # "synthesizer_type": "DataAnalysis"}
    rag_client.patch_config(update_dict)

    if input_file.name.endswith(".csv"):
        df = pd.read_csv(input_file.name)
    elif input_file.name.endswith(".xlsx"):
        df = pd.read_excel(input_file.name)
    else:
        return "Unsupported file type."

    return df.head(10)  # .to_markdown()


def respond(question, chatbot):
    update_dict = {
        "retrieval_mode": "data_analysis",
        "synthesizer_type": "DataAnalysis",
    }
    rag_client.patch_config(update_dict)

    response_gen = rag_client.query(question, stream=True)
    content = ""
    chatbot.append((question, content))
    for resp in response_gen:
        chatbot[-1] = (question, resp.result)
        yield chatbot


def clear_history(chatbot):
    chatbot = []
    global current_session_id
    current_session_id = None
    return chatbot


def reset_textbox():
    return gr.update(value="")


def create_data_analysis_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=4):
            upload_file = gr.File(
                label="Upload csv/xlsx file for data analysis",
                file_count="single",
                file_types=[".xlsx", ".csv"],
                elem_id="upload_file",
                scale=8,
            )
            with gr.Row():
                with gr.Column(scale=4):
                    output_text = gr.DataFrame(
                        label="Data Preview",
                        value=pd.DataFrame(),
                        visible=True,
                        scale=10,
                    )
                #     upload_file_state_df = gr.DataFrame(
                #     label="Upload Status Info", visible=False
                # )

            upload_file.upload(
                fn=upload_file_fn,
                inputs=upload_file,
                outputs=output_text,
                api_name="upload_file_fn",
            )

            # vec_args = {
            #     retrieval_mode,
            # }

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=500, elem_id="data_analysis_chatbot")
            question = gr.Textbox(label="Enter your question.", elem_id="question")
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                clearBtn = gr.Button("Clear History", variant="secondary")

        # TODO，逻辑更新
        # chat_args = (
        #     {question, chatbot}
        #     .union(vec_args)
        #     # .union(llm_args)
        # )

        submitBtn.click(
            fn=respond,
            inputs=[question, chatbot],
            outputs=[chatbot],
            api_name="respond_clk",
        )

        # 绑定Textbox提交事件，当按下Enter，调用respond函数
        question.submit(
            respond,
            inputs=[question, chatbot],
            outputs=[chatbot],
            api_name="respond_q",
        )

        submitBtn.click(
            fn=reset_textbox,
            inputs=[],
            outputs=[question],
            api_name="reset_clk",
        )
        question.submit(
            fn=reset_textbox,
            inputs=[],
            outputs=[question],
            api_name="reset_q",
        )
        clearBtn.click(
            fn=clear_history,
            inputs=[chatbot],
            outputs=[chatbot],
            api_name="clear_history",
        )

        return {
            upload_file.elem_id: upload_file,
        }