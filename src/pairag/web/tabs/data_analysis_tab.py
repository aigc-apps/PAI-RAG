from typing import Dict, Any, List
import gradio as gr
import pandas as pd
import datetime
from pairag.web.rag_local_client import rag_client, RagApiError
from pairag.web.ui_constants import (
    NL2SQL_GENERAL_PROMPTS,
    SYN_GENERAL_PROMPTS,
)
from loguru import logger


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


def upload_history_fn(json_file, database):
    if json_file is None:
        return None
    try:
        # 调用接口
        res = rag_client.add_db_history(json_file.name, database)
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


def upload_description_fn(input_files: List, database):
    if not input_files:
        return None
    try:
        # 调用接口
        res = rag_client.add_db_description(
            [file.name for file in input_files], database
        )
        # yield gr.update(visible=True, value="Upload successfully!")

        # 更新config
        update_dict = {
            "database_file_path": res["destination_path"],
        }
        rag_client.patch_config(update_dict)

        return "Upload successfully!"

    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")


async def load_db_info_fn(input_elements: List[Any]):
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
        rag_client.patch_config(update_dict)
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")

    try:
        await rag_client.load_db_info()
        return f"[{datetime.datetime.now()}] DB info loaded successfully!"
    except RagApiError as api_error:
        raise gr.Error(f"HTTP {api_error.code} Error: {api_error.msg}")
        # return f"[{datetime.datetime.now()}] DB info loaded failed, HTTP {api_error.code} Error: {api_error.msg}"


async def respond(input_elements: List[Any]):
    update_dict = {}
    for element, value in input_elements.items():
        update_dict[element.elem_id] = value

    # 过滤 chatbot 内容
    chatbot = update_dict["chatbot"]
    filtered_chatbot = _filter_chatbot(chatbot)
    # 更新 update_dict 中的 chatbot
    update_dict["chatbot"] = filtered_chatbot

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
    da_chat_model_id = update_dict["data_analysis_model_id"]
    # if not update_dict["include_history"]:
    #     chatbot = clear_history(chatbot)

    q_msg = {"content": question, "role": "user"}
    chatbot.append(q_msg)

    if chatbot is not None:
        chatbot.append(
            {"content": "", "role": "assistant", "metadata": {"status": "pending"}}
        )
        yield chatbot

    try:
        # print(chatbot)
        response_gen = rag_client.query(
            chat_messages=chatbot[:-1],
            stream=True,
            chat_model_id=da_chat_model_id,
            chat_db=True,
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


def _extract_core_content(content: str) -> str:
    """
    从 content 中去掉参考资料部分。
    """
    # 找到“参考资料”开始的位置
    ref_start = content.find("**参考资料**")
    if ref_start != -1:
        return content[:ref_start].strip()  # 截取参考资料之前的内容
    return content.strip()


def _filter_chatbot(chatbot: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    遍历 chatbot 列表，提取每条消息的核心内容，去掉参考资料部分。
    """
    filtered_chatbot = []
    if not chatbot:
        return chatbot
    for message in chatbot:
        if message.get("role", "") == "assistant":
            # 提取核心内容，避免参考资料的信息干扰
            core_content = _extract_core_content(message.get("content", ""))
            # 构造新的消息字典
            filtered_message = {
                "content": core_content,
                "role": message.get("role", ""),
                "metadata": message.get("metadata", {}),
                "options": message.get("options", None),
            }
            filtered_chatbot.append(filtered_message)
        else:
            filtered_chatbot.append(message)

    return filtered_chatbot


def clear_history(chatbot):
    # rag_client.clear_history()
    chatbot = []
    return chatbot


def reset_textbox():
    return gr.update(value="")


# 处理description复选框变化
def handle_description_checkbox_change(db_description_upload):
    if db_description_upload:
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


# 处理history复选框变化
def handle_history_checkbox_change(enable_db_history):
    if enable_db_history:
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


# 处理embedding复选框变化
def handle_embedding_checkbox_change(enable_db_embedding):
    if enable_db_embedding:
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


def create_data_analysis_tab() -> Dict[str, Any]:
    rag_config = rag_client.get_config()
    model_choices = [
        llm.model_id if llm.model_id else llm.model for llm in rag_config.llms
    ]
    model_name = rag_config.data_analysis.model_id
    if not model_name:
        if len(model_choices) == 0:
            model_name = ""
        else:
            model_name = model_choices[0]
    with gr.Row():
        with gr.Column(scale=4):
            data_analysis_model_id = gr.Dropdown(
                choices=model_choices,
                value=model_name,
                label="对话模型ID",
                elem_id="data_analysis_model_id",
            )
            data_analysis_type = gr.Radio(
                choices=[
                    "datafile",
                    "database",
                ],
                value="database",
                label="请选择待分析的数据类型",
                elem_id="analysis_type",
            )

            # datafile
            with gr.Column(
                visible=(data_analysis_type.value == "datafile")
            ) as file_col:
                upload_file = gr.File(
                    label="上传 csv/xlsx 文件以进行数据分析",
                    file_count="single",
                    file_types=[".xlsx", ".csv"],
                    elem_id="upload_file",
                    scale=8,
                )
                output_text = gr.DataFrame(
                    label="数据文件预览",
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
                        label="数据库类型", elem_id="db_dialect", value="mysql"
                    )
                    port = gr.Textbox(label="数据库端口号", elem_id="db_port", value=3306)
                    host = gr.Textbox(label="数据库主机地址", elem_id="db_host")
                with gr.Row():
                    user = gr.Textbox(label="用户名", elem_id="db_username")
                    password = gr.Textbox(
                        label="密码", elem_id="db_password", type="password"
                    )
                with gr.Row():
                    database = gr.Textbox(label="数据库名称", elem_id="database")
                    tables = gr.Textbox(
                        label="数据表名称",
                        elem_id="db_tables",
                        placeholder="列出有用的表名，用逗号分隔，例如：表_A, 表_B, ...，如果为空则使用所有表",
                    )
                descriptions = gr.Textbox(
                    label="表的描述性信息",
                    lines=2,
                    elem_id="db_descriptions",
                    placeholder='一个表描述的字典，例如：{"table_A": "comment_A", "table_B": "comment_B"}',
                )

                with gr.Column(visible=True):
                    enhance_argument = gr.Accordion("大型数据库的增强方案", open=False)
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
                                    info="通过向量嵌入技术优化数据库检索",
                                    elem_id="enable_db_embedding",
                                )

                                max_column_num = gr.Slider(
                                    minimum=50,
                                    maximum=200,
                                    step=10,
                                    label="最大列数",
                                    info="提取唯一值的最大列数",
                                    elem_id="max_column_num",
                                    value=100,
                                    visible=False,  # 初始状态为不可见
                                )
                                max_value_num = gr.Slider(
                                    minimum=1000,
                                    maximum=20000,
                                    step=1000,
                                    label="最大唯一值数量",
                                    info="每列嵌入的唯一值的最大数量。数值越大，可能耗费的时间越长。",
                                    elem_id="max_value_num",
                                    value=1000,
                                    visible=False,  # 初始状态为不可见
                                )

                                enable_db_embedding.change(
                                    fn=handle_embedding_checkbox_change,
                                    inputs=[enable_db_embedding],
                                    outputs=[max_column_num, max_value_num],
                                )

                                enable_db_selector = gr.Checkbox(
                                    label="Yes",
                                    info="通过LLM选表选列",
                                    elem_id="enable_db_selector",
                                )

                                enable_db_history = gr.Checkbox(
                                    label="Yes",
                                    info="使用数据库查询历史/示例",
                                    elem_id="enable_db_history",
                                )

                                db_description_upload = gr.Checkbox(
                                    label="Yes",
                                    info="上传数据表描述信息",
                                    elem_id="enable_db_description_upload",
                                )

                                history_file_upload = gr.File(
                                    label="上传 Q-SQL JSON 文件",
                                    file_count="single",
                                    file_types=[".json"],
                                    elem_id="query_history_upload",
                                    visible=False,  # 初始状态为不可见
                                )

                                history_update_state = gr.Textbox(
                                    label="JSON文件上传状态",
                                    container=False,
                                    visible=False,  # 初始状态为不可见
                                )

                                db_description_file_upload = gr.File(
                                    label="上传数据库描述文件",
                                    file_count="multiple",
                                    file_types=[".csv"],
                                    elem_id="db_description_file_upload",
                                    scale=6,
                                    visible=False,  # 初始状态为不可见
                                )
                                description_update_state = gr.Textbox(
                                    label="数据库描述文件上传状态",
                                    visible=False,  # 初始状态为不可见
                                    container=False,
                                )

                                # 当复选框状态变化时，调用 handle_checkbox_change 函数
                                db_description_upload.change(
                                    fn=handle_description_checkbox_change,
                                    inputs=[db_description_upload],
                                    outputs=[
                                        db_description_file_upload,
                                        description_update_state,
                                    ],
                                )

                                db_description_file_upload.upload(
                                    fn=upload_description_fn,
                                    inputs=[db_description_file_upload, database],
                                    outputs=description_update_state,
                                    api_name="upload_description_fn",
                                )

                                # 当复选框状态变化时，调用 handle_checkbox_change 函数
                                enable_db_history.change(
                                    fn=handle_history_checkbox_change,
                                    inputs=[enable_db_history],
                                    outputs=[history_file_upload, history_update_state],
                                )

                                history_file_upload.upload(
                                    fn=upload_history_fn,
                                    inputs=[history_file_upload, database],
                                    outputs=history_update_state,
                                    api_name="upload_history_fn",
                                )

                # load db info
                with gr.Row():
                    load_db_info_btn = gr.Button(
                        value="加载数据库信息", variant="primary", scale=4
                    )
                    save_state = gr.Textbox(label="数据库加载状态: ", container=False, scale=6)

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
                    with gr.Tab("Nl2sql 提示词模板"):
                        db_nl2sql_prompt = gr.Textbox(
                            label="nl2sql 提示词模板",
                            elem_id="db_nl2sql_prompt",
                            value="",
                            lines=6,
                        )

                    with gr.Tab("合成器 提示词模板"):
                        synthesizer_prompt = gr.Textbox(
                            label="合成器 提示词模板",
                            elem_id="synthesizer_prompt",
                            value="",
                            lines=6,
                        )
                    with gr.Tab("提示词模板重置"):
                        reset_nl2sql_prompt_btn = gr.Button("重置 Nl2sql 提示词模板")
                        reset_synthesizer_prompt_btn = gr.Button("重置 合成器 提示词模板")

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
            chatbot = gr.Chatbot(height=600, elem_id="chatbot", type="messages")
            # with gr.Row():
            #     include_history = gr.Checkbox(
            #         label="Chat history",
            #         info="Query with chat history.",
            #         elem_id="include_history",
            #         value=False,
            #         scale=1,
            #     )
            #     question = gr.Textbox(
            #         label="Enter your question.", elem_id="question", scale=9
            #     )
            question = gr.Textbox(label="在这里输入您的问题", elem_id="question")
            with gr.Row():
                submitBtn = gr.Button("提交", variant="primary")
                clearBtn = gr.Button("清空历史", variant="secondary")

        chat_args = {
            data_analysis_model_id,
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
            # include_history,
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
            data_analysis_model_id.elem_id: data_analysis_model_id,
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
