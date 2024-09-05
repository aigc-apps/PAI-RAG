import os
import json
import datetime
import re
from typing import Dict, Any, List
import gradio as gr
import pandas as pd
from pai_rag.app.web.rag_client import rag_client, RagApiError


DEFAULT_IS_INTERACTIVE = os.environ.get("PAIRAG_RAG__SETTING__interactive", "true")


def upload_file_fn(input_file):
    if input_file is None:
        return None
    try:
        # 调用接口
        res = rag_client.add_datasheet(input_file.name)
        # 更新config
        update_dict = {
            "analysis_type": "nl2pandas",
            "analysis_file_path": res["destination_path"],
        }
        rag_client.patch_config(update_dict)

        # json_str = res["data_preview"]

        # # 将json字符串加载为列表
        # # data_list = json.loads(json_str)
        # # # 将列表转换为 DataFrame
        # # df = pd.DataFrame(data_list)

        # df = pd.read_json(json_str)

        if input_file.name.endswith(".csv"):
            df = pd.read_csv(input_file.name)
            return df.head(10)
        elif input_file.name.endswith(".xlsx"):
            df = pd.read_excel(input_file.name)
            return df.head(10)
        else:
            return "Unsupported file type."
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def connect_database(input_db: List[Any]):
    try:
        update_dict = {"analysis_type": "nl2sql"}
        for element, value in input_db.items():
            if (element.elem_id == "db_tables") and (value != ""):
                # 去掉首位空格和末尾逗号
                value = value.strip().rstrip(",")
                # 英文逗号和中文逗号作为分隔符进行分割，并去除多余空白字符
                value = [word.strip() for word in re.split(r"\s*,\s*|，\s*", value)]
                # 检查是否为列表
                if isinstance(value, list):
                    print(f"Valid input: {value}")
                else:
                    return "Invalid input: Input must be table_A, table_B,..."
            if (element.elem_id == "db_descriptions") and (value != ""):
                value = json.loads(value)
                # 检查是否为字典
                if isinstance(value, dict):
                    print(f"Valid input: {value}")
                else:
                    return "Invalid input: Input must be a dictionary."
            update_dict[element.elem_id] = value
        # print("db_config:", update_dict)

        rag_client.patch_config(update_dict)
        return f"[{datetime.datetime.now()}] Connect database success!"
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def analysis_respond(question, chatbot):
    response_gen = rag_client.query_data_analysis(question, stream=True)
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
            data_analysis_type = gr.Radio(
                choices=[
                    "datafile",
                    "database",
                ],
                value="datafile",
                label="Please choose data analysis type",
                elem_id="data_analysis_type",
            )

            # datafile
            with gr.Column(
                visible=(data_analysis_type.value == "datafile")
            ) as file_col:
                upload_file = gr.File(
                    label="Upload csv/xlsx file for data analysis",
                    file_count="single",
                    file_types=[".xlsx", ".csv"],
                    elem_id="upload_file",
                    scale=8,
                )
                output_text = gr.DataFrame(
                    label="Data File Preview",
                    value=pd.DataFrame(),
                    visible=True,
                    scale=10,
                )

            upload_file.upload(
                fn=upload_file_fn,
                inputs=upload_file,
                outputs=output_text,
                api_name="upload_analysis_file_fn",
            )

            # database
            with gr.Column(visible=(data_analysis_type.value == "database")) as db_col:
                dialect = gr.Textbox(
                    label="Dialect", elem_id="db_dialect", value="mysql"
                )
                user = gr.Textbox(label="Username", elem_id="db_username")
                password = gr.Textbox(
                    label="Password", elem_id="db_password", type="password"
                )
                host = gr.Textbox(label="Host", elem_id="db_host")
                port = gr.Textbox(label="Port", elem_id="db_port", value=3306)
                dbname = gr.Textbox(label="DBname", elem_id="db_name")
                tables = gr.Textbox(
                    label="Tables",
                    elem_id="db_tables",
                    placeholder="List db tables, separated by commas, e.g. table_A, table_B, ... , using all tables if blank",
                )
                descriptions = gr.Textbox(
                    label="Descriptions",
                    lines=5,
                    elem_id="db_descriptions",
                    placeholder="A dict of table descriptions, e.g. {'table_A': 'text_description_A', 'table_B': 'text_description_B'}",
                )

                connect_db_button = gr.Button(
                    "Connect Database",
                    elem_id="connect_db_button",
                    variant="primary",
                )  # 点击功能中更新analysis_type

                connection_info = gr.Textbox(
                    label="Connection Info", elem_id="db_connection_info"
                )

            inputs_db = {
                dialect,
                user,
                password,
                host,
                port,
                dbname,
                tables,
                descriptions,
            }

            connect_db_button.click(
                fn=connect_database,
                inputs=inputs_db,
                outputs=connection_info,
                api_name="connect_db",
            )

            def data_analysis_type_change(type_value):
                if type_value == "datafile":
                    return {
                        file_col: gr.update(visible=type_value),
                        db_col: gr.update(visible=False),
                    }
                elif type_value == "database":
                    return {
                        db_col: gr.update(visible=type_value),
                        file_col: gr.update(visible=False),
                    }

            data_analysis_type.change(
                fn=data_analysis_type_change,
                inputs=data_analysis_type,
                outputs=[file_col, db_col],
            )

        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=500, elem_id="data_analysis_chatbot")
            question = gr.Textbox(label="Enter your question.", elem_id="question")
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                clearBtn = gr.Button("Clear History", variant="secondary")

        submitBtn.click(
            fn=analysis_respond,
            inputs=[question, chatbot],
            outputs=[chatbot],
            api_name="analysis_respond_clk",
        )

        # 绑定Textbox提交事件，当按下Enter，调用respond函数
        question.submit(
            analysis_respond,
            inputs=[question, chatbot],
            outputs=[chatbot],
            api_name="analysis_respond_q",
        )

        submitBtn.click(
            fn=reset_textbox,
            inputs=[],
            outputs=[question],
            api_name="analysis_reset_clk",
        )
        question.submit(
            fn=reset_textbox,
            inputs=[],
            outputs=[question],
            api_name="analysis_reset_q",
        )
        clearBtn.click(
            fn=clear_history,
            inputs=[chatbot],
            outputs=[chatbot],
            api_name="analysi_clear_history",
        )

        return {
            upload_file.elem_id: upload_file,
            dialect.elem_id: dialect,
            user.elem_id: user,
            password.elem_id: password,
            host.elem_id: host,
            port.elem_id: port,
            dbname.elem_id: dbname,
            tables.elem_id: tables,
            descriptions.elem_id: descriptions,
        }
