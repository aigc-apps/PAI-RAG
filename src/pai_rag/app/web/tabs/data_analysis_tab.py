from typing import Dict, Any, List
import gradio as gr
import pandas as pd
import datetime
from pai_rag.app.web.rag_client import rag_client, RagApiError
from pai_rag.app.web.ui_constants import (
    NL2SQL_GENERAL_PROMPTS,
    SYN_GENERAL_PROMPTS,
)


def upload_file_fn(input_file):
    if input_file is None:
        return None
    try:
        # 调用接口
        res = rag_client.add_datasheet(input_file.name)
        # 更新config
        update_dict = {
            "analysis_type": "pandas",
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


def load_db_info_fn(input_elements: List[Any]):
    update_dict = {}
    for element, value in input_elements.items():
        update_dict[element.elem_id] = value

    # update snapshot
    try:
        update_dict["analysis_type"] = "nl2sql"
        if update_dict["enable_db_embedding"] is True:
            update_dict["enable_query_preprocessor"] = True
            update_dict["enable_db_preretriever"] = True
        else:
            update_dict["enable_query_preprocessor"] = False
            update_dict["enable_db_preretriever"] = False
        print("update_dict:", update_dict)
        rag_client.patch_config(update_dict)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    try:
        rag_client.load_db_info()
        return f"[{datetime.datetime.now()}] DB info loaded successfully!"
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")
        # return f"[{datetime.datetime.now()}] DB info loaded failed, HTTP {api_error.code} Error: {api_error.msg}"


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
        # print("respond udpate_dict:", update_dict)
        rag_client.patch_config(update_dict)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    question = update_dict["question"]
    chatbot = update_dict["chatbot"]

    if chatbot is not None:
        chatbot.append((question, ""))

    try:
        response_gen = rag_client.query_data_analysis(question, stream=False)
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


# 处理history复选框变化
def handle_history_checkbox_change(enable_db_history):
    if enable_db_history:
        return gr.File.update(visible=True), gr.Textbox.update(visible=True)
    else:
        return gr.File.update(visible=False), gr.Textbox.update(visible=False)


# 处理embedding复选框变化
def handle_embedding_checkbox_change(enable_db_embedding):
    if enable_db_embedding:
        return gr.Slider.update(visible=True), gr.Slider.update(visible=True)
    else:
        return gr.Slider.update(visible=False), gr.Slider.update(visible=False)


def upload_history_fn(json_file):
    if json_file is None:
        return None
    try:
        # 调用接口
        res = rag_client.add_db_history(json_file.name)
        # 更新config
        update_dict = {
            "db_history_file_path": res["destination_path"],
        }
        rag_client.patch_config(update_dict)

        if json_file.name.endswith(".json"):
            return "Upload successfully!"
        else:
            return "Please upload a json file."

    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


def create_data_analysis_tab() -> Dict[str, Any]:
    with gr.Row():
        with gr.Column(scale=4):
            data_analysis_type = gr.Radio(
                choices=[
                    "datafile",
                    "database",
                ],
                value="database",
                label="Please choose the analysis type",
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
                    database = gr.Textbox(label="Database", elem_id="database")
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

                with gr.Column(visible=True):
                    enhance_argument = gr.Accordion(
                        "Enhancement options for larger database", open=False
                    )
                    with enhance_argument:
                        with gr.Row():
                            with gr.Column(scale=1):
                                # enable_enhanced_description = gr.Checkbox(
                                #     label="Yes",
                                #     info="Enhance db description by llm",
                                #     elem_id="enable_enhanced_description",
                                # )
                                enable_db_embedding = gr.Checkbox(
                                    label="Yes",
                                    info="Enhance db retrieval by embedding",
                                    elem_id="enable_db_embedding",
                                )

                                max_column_num = gr.Slider(
                                    minimum=50,
                                    maximum=200,
                                    step=10,
                                    label="Max Column Number",
                                    info="Max number of columns to extract unique values.",
                                    elem_id="max_column_num",
                                    value=100,
                                    visible=False,  # 初始状态为不可见
                                )
                                max_value_num = gr.Slider(
                                    minimum=5000,
                                    maximum=20000,
                                    step=1000,
                                    label="Max Value Number",
                                    info="Maximum number of unique values to be embedded. Larger number may take longer time.",
                                    elem_id="max_value_num",
                                    value=10000,
                                    visible=False,  # 初始状态为不可见
                                )

                                enable_db_embedding.change(
                                    fn=handle_embedding_checkbox_change,
                                    inputs=[enable_db_embedding],
                                    outputs=[max_column_num, max_value_num],
                                )

                                enable_db_selector = gr.Checkbox(
                                    label="Yes",
                                    info="Enable db schema selection by llm",
                                    elem_id="enable_db_selector",
                                )

                                enable_db_history = gr.Checkbox(
                                    label="Yes",
                                    info="Enable db query history/example",
                                    elem_id="enable_db_history",
                                )

                                history_file_upload = gr.File(
                                    label="Upload q-sql json file",
                                    file_count="single",
                                    file_types=[".json"],
                                    elem_id="query_history_upload",
                                    visible=False,  # 初始状态为不可见
                                )

                                history_update_state = gr.Textbox(
                                    label="History upload state",
                                    container=False,
                                    visible=False,  # 初始状态为不可见
                                )

                                # 当复选框状态变化时，调用 handle_checkbox_change 函数
                                enable_db_history.change(
                                    fn=handle_history_checkbox_change,
                                    inputs=[enable_db_history],
                                    outputs=[history_file_upload, history_update_state],
                                )

                                history_file_upload.upload(
                                    fn=upload_history_fn,
                                    inputs=history_file_upload,
                                    outputs=history_update_state,
                                    api_name="upload_history_fn",
                                )

                # load db info
                with gr.Row():
                    load_db_info_btn = gr.Button(
                        value="Load DB Info", variant="primary", scale=4
                    )
                    save_state = gr.Textbox(
                        label="DB Load Info: ", container=False, scale=6
                    )

                load_args = {
                    data_analysis_type,
                    dialect,
                    user,
                    password,
                    host,
                    port,
                    database,
                    tables,
                    descriptions,
                    # enable_enhanced_description,
                    enable_db_history,
                    enable_db_embedding,
                    max_column_num,
                    max_value_num,
                    # enable_query_preprocessor,
                    # enable_db_preretriever,
                    enable_db_selector,
                }

                load_db_info_btn.click(
                    fn=load_db_info_fn,
                    inputs=load_args,
                    outputs=[save_state],
                )

                with gr.Column(visible=True):
                    with gr.Tab("Nl2sql Prompt"):
                        db_nl2sql_prompt = gr.Textbox(
                            label="nl2sql template",
                            elem_id="db_nl2sql_prompt",
                            value=NL2SQL_GENERAL_PROMPTS,
                            lines=6,
                        )

                    with gr.Tab("Synthesizer Prompt"):
                        synthesizer_prompt = gr.Textbox(
                            label="synthesizer template",
                            elem_id="synthesizer_prompt",
                            value=SYN_GENERAL_PROMPTS,
                            lines=6,
                        )
                    with gr.Tab("Prompt Reset"):
                        reset_nl2sql_prompt_btn = gr.Button("Reset Nl2sql Prompt")
                        reset_synthesizer_prompt_btn = gr.Button(
                            "Reset Synthesizer Prompt"
                        )

                    def reset_nl2sql_prompt():
                        return gr.update(value=NL2SQL_GENERAL_PROMPTS)

                    def reset_synthesizer_prompt():
                        return gr.update(value=SYN_GENERAL_PROMPTS)

                    reset_nl2sql_prompt_btn.click(
                        fn=reset_nl2sql_prompt,
                        inputs=[],
                        outputs=[db_nl2sql_prompt],
                        api_name="reset_nl2sql_prompt_clk",
                    )
                    reset_synthesizer_prompt_btn.click(
                        fn=reset_synthesizer_prompt,
                        inputs=[],
                        outputs=[synthesizer_prompt],
                        api_name="reset_synthesizer_prompt_clk",
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
            chatbot = gr.Chatbot(height=600, elem_id="chatbot")
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
            database,
            tables,
            descriptions,
            enable_db_selector,
            db_nl2sql_prompt,
            synthesizer_prompt,
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
            database.elem_id: database,
            tables.elem_id: tables,
            descriptions.elem_id: descriptions,
            # enable_enhanced_description.elem_id: enable_enhanced_description,
            enable_db_history.elem_id: enable_db_history,
            enable_db_embedding.elem_id: enable_db_embedding,
            max_column_num.elem_id: max_column_num,
            max_value_num.elem_id: max_value_num,
            # enable_query_preprocessor.elem_id: enable_query_preprocessor,
            # enable_db_preretriever.elem_id: enable_db_preretriever,
            enable_db_selector.elem_id: enable_db_selector,
            db_nl2sql_prompt.elem_id: db_nl2sql_prompt,
            synthesizer_prompt.elem_id: synthesizer_prompt,
        }
