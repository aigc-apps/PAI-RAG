import json
import shutil
from typing import Any, Dict, List
import pandas as pd
import os
import re
import markdown
import html
from loguru import logger
from openai.types.chat import (
    ChatCompletion,
)
from pairag.chat.models import (
    ChatCompletionRequest,
    RetrievalRequest,
)
from pairag.web.view_model import ViewModel
from pairag.web.ui_constants import EMPTY_KNOWLEDGEBASE_MESSAGE
from pairag.core.rag_config import RagConfig
from pairag.knowledgebase.rag_knowledgebase import knowledgebase_manager, KnowledgeBase
from pairag.data_pipeline.job.rag_job_manager import job_manager
from pairag.core.chat_service import chat_service
from datetime import datetime

from pairag.integrations.data_analysis.text2sql.utils.constants import (
    DEFAULT_DESCRIPTION_FOLDER_PATH,
    DEFAULT_DB_HISTORY_PATH,
    DEFAULT_DB_HISTORY_NAME,
)


class RagApiError(Exception):
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_ts():
    dt = datetime.now()
    ms = dt.microsecond // 1000
    return datetime.now().strftime("%Y%m%d%H%M%S") + f"{ms:03d}"


DEFAULT_CLIENT_TIME_OUT = 120
DEFAULT_LOCAL_URL = "http://127.0.0.1:8680/"


class RagLocalClient:
    def _format_rag_response(self, response):
        text = response["delta"]
        docs = response.get("docs", []) or []
        is_finished = response.get("is_finished", True)

        referenced_docs = ""
        if is_finished:
            content_list = []
            for i, doc in enumerate(docs):
                metadata = doc.get("metadata", {})
                filename = metadata.get("name")
                sheet_name = metadata.get("sheet_name")
                ref_table = metadata.get("query_tables")
                invalid_flag = metadata.get("invalid_flag", 0)
                file_url = metadata.get("file_url")
                if doc.get("image_url"):
                    media_url = doc.get("image_url")
                else:
                    media_url = None
                    media_urls = metadata.get("image_info_list")
                    if media_urls:
                        media_url = media_urls[0]
                if media_url and doc.get("text") == "":
                    formatted_image_name = re.sub(
                        "^[0-9a-z]{32}_", "", "/".join(media_url.split("/")[-2:])
                    )
                    content = f"""
<span>
    <a href="{media_url}"> [{i+1}]: {formatted_image_name} </a> 分数:{doc.get("score")}
</span>
<br>
"""
                elif filename:
                    formatted_file_name = re.sub("^[0-9a-z]{32}_", "", filename)
                    if sheet_name:
                        formatted_file_name += f">>{sheet_name}"
                    html_content = html.escape(
                        re.sub(r"<.*?>", "", doc.get("text"))
                    ).replace("\n", " ")
                    if file_url:
                        formatted_file_name = (
                            f'<a href="{file_url}"> {formatted_file_name} </a>'
                        )
                    content = f"""
<span class="text">
    [{i+1}]: {formatted_file_name} 分数:{doc.get("score")}
    <span style='color: gray; font-size: 12px;'> ( {html_content} ) </span>
</span>
<br>
"""
                elif ref_table:
                    ref_table_format = ", ".join([i for i in ref_table])
                    formatted_table_name = f"查询数据库中相关表名包括： <b>{ref_table_format}</b>"

                    if invalid_flag == 0:
                        run_flag = " ✓ "
                        ref_sql = metadata.get("query_code_instruction", None)
                        formatted_sql_query = f"<b>{ref_sql}</b>"
                        content = (
                            f"""<span style="color:grey; font-size: 14px;">{formatted_table_name}</span> \n"""
                            f"""<span style="color:grey; font-size: 14px;">生成的sql语句为：</span> <pre style="color:grey; font-size: 12px;">{formatted_sql_query}</pre> """
                            f"""<span style="color:grey; font-size: 14px;">sql查询是否有效：</span> <span style="color:green; font-size: 14px;">{run_flag}</span>"""
                        )
                    else:
                        run_flag = " ✗ "
                        ref_sql = metadata.get("generated_query_code_instruction", None)
                        formatted_sql_query = f"<b>{ref_sql}</b>"
                        content = (
                            f"""<span style="color:grey; font-size: 14px;">{formatted_table_name}</span> \n"""
                            f"""<span style="color:grey; font-size: 14px;">生成的sql语句为：</span> <pre style="color:grey; font-size: 12px;">{formatted_sql_query}</pre> """
                            f"""<span style="color:grey; font-size: 14px;">sql查询是否有效：</span> <span style="color:red; font-size: 14px;">{run_flag}</span>"""
                        )
                else:
                    content = ""
                content_list.append(content)
            referenced_docs = "".join(content_list)

        formatted_answer = text
        if referenced_docs:
            formatted_answer += f"\n\n**参考资料**:\n {referenced_docs}"

        response["delta"] = formatted_answer

        return dotdict(response)

    def _format_rag_response_v1_chat_completions(self, response):
        text = response["delta"]
        docs = response.get("docs", []) or []
        is_finished = response.get("is_finished", True)

        referenced_docs = ""
        if is_finished:
            content_list = []
            for i, doc in enumerate(docs):
                filename = doc.get("name")
                doc_text = doc.get("text")
                score = doc.get("score")
                url = doc.get("url", "")
                if url and url.startswith("http"):
                    filename = f'<a href="{url}"> {filename} </a>'
                content = f"""
<span class="text">
    [{i+1}]: {filename} 分数:{score}
    <span style='color: gray; font-size: 12px;'> ( {doc_text} ) </span>
</span>
<br>
"""
                content_list.append(content)
            referenced_docs = "".join(content_list)

        formatted_answer = text
        if referenced_docs:
            formatted_answer += f"\n\n**参考资料**:\n {referenced_docs}"

        response["delta"] = formatted_answer

        return dotdict(response)

    async def query(
        self,
        chat_messages: List[Dict[str, str]],
        stream: bool = False,
        citation: bool = False,
        index_name: str = None,
        return_reference: bool = False,
        chat_model_id: str = None,
        temperature: float = 0.1,
        chat_knowledgebase: bool = False,
        search_web: bool = False,
        chat_llm: bool = False,
        chat_db: bool = False,
        chat_news: bool = False,
    ):
        query = ChatCompletionRequest(
            model=chat_model_id,
            messages=chat_messages,
            temperature=temperature,
            stream=stream,
            index_name=index_name,
            citation=citation,
            return_reference=return_reference,
            chat_knowledgebase=chat_knowledgebase,
            search_web=search_web,
            chat_llm=chat_llm,
            chat_db=chat_db,
            chat_news=chat_news,
        )

        try:
            if stream:
                response = await chat_service.astream_chat(query)
            else:
                response = await chat_service.achat(query)

            if isinstance(response, ChatCompletion):
                result = {
                    "delta": response.choices[0].message.content,
                    "docs": response.citation_details,
                    "is_finished": response.choices[0].finish_reason in ["stop", ""],
                }
                yield self._format_rag_response_v1_chat_completions(result)
            else:
                async for r in response:
                    if r.startswith("data: "):
                        chunk = json.loads(r[6:])
                        result = {
                            "delta": chunk["choices"][0]["delta"]["content"],
                            "docs": chunk.get("citation_details", []),
                            "is_finished": chunk["choices"][0]["finish_reason"]
                            == "stop",
                        }
                        if chat_knowledgebase or search_web:
                            yield self._format_rag_response_v1_chat_completions(result)
                        else:
                            yield self._format_rag_response(result)
        except Exception as e:
            raise RagApiError(code=500, msg=str(e))

    def get_upload_history(self, knowledgebase_name):
        return job_manager.get_job_history(name=knowledgebase_name)

    def add_datasheet(
        self,
        input_file: str,
    ):
        if not input_file:
            return None

        persist_path = "./localdata/data_analysis"
        os.makedirs(name=persist_path, exist_ok=True)

        # 清空目录中的文件
        for filename in os.listdir(persist_path):
            file_path = os.path.join(persist_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.info(f"Failed to delete {file_path}. Reason: {e}")

        # 指定持久化存储位置
        file_name = os.path.basename(input_file)  # 获取文件名
        destination_path = os.path.join(persist_path, file_name)
        # 写入文件
        try:
            shutil.copy(input_file, destination_path)
            logger.info(f"data analysis file saved successfully to {destination_path}.")

            if destination_path.endswith(".csv"):
                df = pd.read_csv(destination_path)
            elif destination_path.endswith(".xlsx"):
                df = pd.read_excel(destination_path)
            else:
                raise TypeError("Unsupported file type.")

        except Exception as e:
            raise RagApiError(code=500, msg=str(e))

        return {
            "destination_path": destination_path,
            "data_preview": df.head(10).to_json(orient="records", lines=False),
        }

    def add_db_history(
        self,
        input_file: str,
        db_name: str,
    ):
        if not input_file:
            return None

        persist_path = DEFAULT_DB_HISTORY_PATH
        os.makedirs(name=persist_path, exist_ok=True)

        # 指定持久化存储位置
        file_name = os.path.basename(input_file)  # 获取文件名
        destination_path = os.path.join(persist_path, f"{db_name}_{file_name}")
        # 写入文件
        try:
            # shutil.copy(file.filename, destination_path)
            # with open(destination_path, "wb") as f:
            #     shutil.copyfileobj(input_file, f)
            shutil.copy(input_file, destination_path)
            logger.info("History file saved successfully")

        except Exception as e:
            raise RagApiError(code=500, msg=str(e))

        # 重命名
        try:
            unified_destination_path = os.path.join(
                persist_path, f"{db_name}_{DEFAULT_DB_HISTORY_NAME}"
            )
            os.rename(destination_path, unified_destination_path)
            logger.info("History file renamed successfully")
        except Exception as e:
            raise RagApiError(code=500, msg=str(e))

        return {
            "destination_path": unified_destination_path,
        }

    def add_db_description(
        self,
        files: List[str],
        db_name: str,
    ):
        if not files:
            return {"message": "No upload files"}

        persist_path = DEFAULT_DESCRIPTION_FOLDER_PATH
        file_destination_folder = os.path.join(
            persist_path, db_name, "database_description"
        )
        os.makedirs(name=file_destination_folder, exist_ok=True)

        # 指定持久化存储位置
        for file in files:
            file_name = os.path.basename(file)  # 获取文件名
            file_destination_path = os.path.join(
                persist_path, db_name, "database_description", file_name
            )
            # 写入文件
            try:
                # shutil.copy(file.filename, destination_path)
                # with open(file_destination_path, "wb") as f:
                #     shutil.copyfileobj(file, f)
                shutil.copy(file, file_destination_path)
                logger.info(
                    f"Description file saved successfully: {file_destination_path}"
                )

            except Exception as e:
                raise RagApiError(code=500, msg=str(e))

        return {
            "destination_path": persist_path,
        }

    async def load_db_info(
        self,
    ):
        try:
            await chat_service.aload_db_info()
        except Exception as e:
            logger.exception(f"load db info failed: {e}")
            raise  # 重新抛出异常

    def get_knowledge_state(self, task_id: str):
        status, detail = chat_service.get_task_status(task_id=task_id)
        return {
            "status": status,
            "detail": detail,
        }

    def patch_config(self, update_dict: Any):
        config = self.get_config()
        # print("config:", config)
        view_model: ViewModel = ViewModel.from_app_config(config)
        view_model.update(update_dict)

        new_config = view_model.to_app_config()
        try:
            config = chat_service.reload(new_config)
            return
        except Exception as e:
            logger.exception(f"patch config failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"patch config failed. {e}",
            )

    def get_config(self):
        try:
            config = chat_service.get_config()
            rag_config = RagConfig.model_validate(config)
            return rag_config

        except Exception as e:
            logger.exception(f"get config failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"get config failed. {e}",
            )

    def list_indexes(self):
        try:
            return knowledgebase_manager.list_knowledgebases()
        except Exception as e:
            logger.exception(f"list index failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"list index failed. {e}",
            )

    def add_index(self, index_entry: KnowledgeBase):
        try:
            knowledgebase_manager.add_knowledgebase(knowledgebase=index_entry)
        except Exception as e:
            logger.exception(f"add index {index_entry.name} failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"add index {index_entry.name} failed. {e}",
            )

    def update_index(self, index_entry: KnowledgeBase):
        try:
            knowledgebase_manager.update_knowledgebase(knowledgebase=index_entry)
        except Exception as e:
            logger.exception(f"update index {index_entry.name} failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"update index {index_entry.name} failed. {e}",
            )

    def update_index_retrieval_settings(
        self,
        knowledgebase_id: str,
        retrieval_settings: dict = {},
    ):
        try:
            _knowledgebase = knowledgebase_manager.get_knowledgebase(knowledgebase_id)
            _knowledgebase.retrieval_settings = retrieval_settings
            knowledgebase_manager.update_knowledgebase(_knowledgebase)
        except Exception as e:
            logger.exception(
                f"update retrieval_settings for index {knowledgebase_id} failed: {e}"
            )
            raise RagApiError(
                code=500,
                msg=f"update retrieval_settings for index {knowledgebase_id} failed. {e}",
            )

    def update_index_qa_prompt_templates(
        self,
        knowledgebase_id: str,
        qa_prompt_templates: dict = {},
    ):
        try:
            _knowledgebase = knowledgebase_manager.get_knowledgebase(knowledgebase_id)
            _knowledgebase.qa_prompt_templates = qa_prompt_templates
            knowledgebase_manager.update_knowledgebase(_knowledgebase)
        except Exception as e:
            logger.exception(
                f"update qa_prompt_templates for index {knowledgebase_id} failed: {e}"
            )
            raise RagApiError(
                code=500,
                msg=f"update qa_prompt_templates for index {knowledgebase_id} failed. {e}",
            )

    def get_index_retrieval_settings(self, knowledgebase_id: str):
        try:
            _knowledgebase = knowledgebase_manager.get_knowledgebase(knowledgebase_id)
            return _knowledgebase.retrieval_settings
        except Exception as e:
            logger.exception(
                f"update retrieval_settings for index {knowledgebase_id} failed: {e}"
            )
            raise RagApiError(
                code=500,
                msg=f"update retrieval_settings for index {knowledgebase_id} failed. {e}",
            )

    def get_index_qa_prompt_templates(self, knowledgebase_id: str):
        try:
            _knowledgebase = knowledgebase_manager.get_knowledgebase(knowledgebase_id)
            return _knowledgebase.qa_prompt_templates
        except Exception as e:
            logger.exception(
                f"Get qa_prompt_templates for index {knowledgebase_id} failed: {e}"
            )
            raise RagApiError(
                code=500,
                msg=f"Get qa_prompt_templates for index {knowledgebase_id} failed. {e}",
            )

    def delete_index(self, index_name: str):
        try:
            knowledgebase_manager.delete_knowledgebase(name=index_name)
        except Exception as e:
            logger.exception(f"delete index {index_name} failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"delete index {index_name} failed. {e}",
            )

    async def aknowledgebase_retrieval(
        self,
        knowledgebase_id: str,
        query: str,
        retrieval_settings: dict = {},
    ):
        try:
            response = await chat_service.aknowledgebase_retrieval(
                RetrievalRequest(
                    query=query,
                    knowledgebase_id=knowledgebase_id,
                    retrieval_settings=retrieval_settings,
                )
            )
            result = {}
            formatted_text = "<tr><th>切片</th><th>分数</th><th>文本</th><th>标题</th></tr>\n"
            if len(response.records) == 0:
                result["delta"] = EMPTY_KNOWLEDGEBASE_MESSAGE.format(query_str=query)
            else:
                for i, record in enumerate(response.records):
                    html_content = markdown.markdown(record.content)
                    file_url = record.metadata.get("file_url", None)
                    safe_html_content = html.escape(html_content).replace("\n", "<br>")
                    if file_url:
                        safe_html_content = (
                            f"""<a href="{file_url}">{safe_html_content}</a>"""
                        )
                    formatted_text += '<tr style="font-size: 13px;"><td>切片 {}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n'.format(
                        i + 1, record.score, safe_html_content, record.title
                    )
                formatted_text = (
                    "<table>\n<tbody>\n" + formatted_text + "</tbody>\n</table>"
                )
                result["delta"] = formatted_text
            yield dotdict(result)

        except Exception as error:
            raise RagApiError(code=500, msg=str(error))

    def get_knowledgebase_retrieval_config(self, knowledgebase_id):
        try:
            config = chat_service.get_config()
            rag_config = RagConfig.model_validate(config)
            return rag_config

        except Exception as e:
            logger.exception(f"get config failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"get config failed. {e}",
            )


rag_client = RagLocalClient()
