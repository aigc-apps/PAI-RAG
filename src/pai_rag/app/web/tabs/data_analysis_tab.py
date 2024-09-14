import os
from typing import Dict, Any, List
import gradio as gr
import pandas as pd
from pai_rag.app.web.rag_client import rag_client, RagApiError
from pai_rag.app.web.ui_constants import DA_GENERAL_PROMPTS, DA_SQL_PROMPTS


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


def respond(input_elements: List[Any]):
    update_dict = {}
    for element, value in input_elements.items():
        update_dict[element.elem_id] = value

    if update_dict["analysis_type"] == "datafile":
        update_dict["analysis_type"] = "nl2pandas"
    else:
        update_dict["analysis_type"] = "nl2sql"

    # empty input.
    if not update_dict["question"]:
        yield update_dict["chatbot"]
        return

    # update snapshot
    try:
        rag_client.patch_config(update_dict)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    question = update_dict["question"]
    chatbot = update_dict["chatbot"]

    if chatbot is not None:
        chatbot.append((question, ""))

    try:
        response_gen = rag_client.query_data_analysis(question, stream=True)
        for resp in response_gen:
            chatbot[-1] = (question, resp.result)
            yield chatbot
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")
    except Exception as e:
        raise gr.Error(f"Error: {e}")
    finally:
        yield chatbot


def clear_history(chatbot):
    rag_client.clear_history()
    chatbot = []
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
                label="Please choose the data analysis type",
                elem_id="analysis_type",
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
                with gr.Row():
                    dialect = gr.Textbox(
                        label="Dialect", elem_id="db_dialect", value="mysql"
                    )
                    port = gr.Textbox(label="Port", elem_id="db_port", value=3306)
                    host = gr.Textbox(label="Host", elem_id="db_host")
                with gr.Row():
                    user = gr.Textbox(label="Username", elem_id="db_username")
                    password = gr.Textbox(
                        label="Password", elem_id="db_password", type="password"
                    )
                with gr.Row():
                    dbname = gr.Textbox(label="DBname", elem_id="db_name")
                    tables = gr.Textbox(
                        label="Tables",
                        elem_id="db_tables",
                        placeholder="List db tables, separated by commas, e.g. table_A, table_B, ... , using all tables if blank",
                    )
                descriptions = gr.Textbox(
                    label="Descriptions",
                    lines=3,
                    elem_id="db_descriptions",
                    placeholder='A dict of table descriptions, e.g. {"table_A": "text_description_A", "table_B": "text_description_B"}',
                )

                prompt_type = gr.Radio(
                    [
                        "general",
                        "sql",
                        "custom",
                    ],
                    value="general",
                    label="\N{rocket} Please choose the prompt template type",
                    elem_id="nl2sql_prompt_type",
                )

                prompt_template = gr.Textbox(
                    label="prompt template",
                    elem_id="db_nl2sql_prompt",
                    value=DA_GENERAL_PROMPTS,
                    lines=4,
                )

            def change_prompt_template(prompt_type):
                if prompt_type == "general":
                    return {
                        prompt_template: gr.update(
                            value=DA_GENERAL_PROMPTS, interactive=False
                        )
                    }
                elif prompt_type == "sql":
                    return {
                        prompt_template: gr.update(
                            value=DA_SQL_PROMPTS, interactive=False
                        )
                    }
                else:
                    return {prompt_template: gr.update(value="", interactive=True)}

            prompt_type.input(
                fn=change_prompt_template,
                inputs=prompt_type,
                outputs=[prompt_template],
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
            chatbot = gr.Chatbot(height=500, elem_id="chatbot")
            question = gr.Textbox(label="Enter your question.", elem_id="question")
            with gr.Row():
                submitBtn = gr.Button("Submit", variant="primary")
                clearBtn = gr.Button("Clear History", variant="secondary")

        chat_args = {
            data_analysis_type,
            dialect,
            user,
            password,
            host,
            port,
            dbname,
            tables,
            descriptions,
            prompt_template,
            question,
            chatbot,
        }

        submitBtn.click(
            fn=respond,
            inputs=chat_args,
            outputs=[chatbot],
            api_name="analysis_respond_clk",
        )

        # 绑定Textbox提交事件，当按下Enter，调用respond函数
        question.submit(
            respond,
            inputs=chat_args,
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
            prompt_type.elem_id: prompt_type,
            prompt_template.elem_id: prompt_template,
        }
