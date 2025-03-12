import asyncio
import hashlib
import json
import shutil
from typing import Any, Dict, List
import uuid
import pandas as pd
import os
import re
import markdown
import html
from loguru import logger
from pai_rag.app.api.models import RagQuery, RagResponse
from pai_rag.app.web.rag_client import RagApiError, dotdict
from pai_rag.app.web.view_model import ViewModel
from pai_rag.app.web.ui_constants import EMPTY_KNOWLEDGEBASE_MESSAGE
from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_index_manager import RagIndexEntry, RagIndexMap, index_manager
from pai_rag.core.rag_service import rag_service
from datetime import datetime
import time
from starlette.concurrency import run_in_threadpool

from pai_rag.integrations.nodeparsers.pai.pai_node_parser import (
    COMMON_FILE_PATH_FODER_NAME,
)
from pai_rag.integrations.data_analysis.text2sql.utils.constants import (
    DEFAULT_DESCRIPTION_FOLDER_PATH,
    DEFAULT_DB_HISTORY_PATH,
    DEFAULT_DB_HISTORY_NAME,
)


def get_ts():
    dt = datetime.now()
    ms = dt.microsecond // 1000
    return datetime.now().strftime("%Y%m%d%H%M%S") + f"{ms:03d}"


def _create_chat_history_from_messages(chat_messages):
    chat_history = []
    for message in chat_messages:
        if message["role"] == "user":
            chat_history.append(
                {
                    "user": str(message["content"]),
                }
            )
        elif message["role"] == "assistant" and len(chat_history) > 0:
            chat_history[-1]["bot"] = str(message["content"])

    return chat_history


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
                filename = metadata.get("file_name")
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
    <a href="{media_url}"> [{i+1}]: {formatted_image_name} </a> Score:{doc.get("score")}
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
    [{i+1}]: {formatted_file_name} Score:{doc.get("score")}
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
            formatted_answer += f"\n\n**Reference**:\n {referenced_docs}"

        response["delta"] = formatted_answer

        return dotdict(response)

    async def query(
        self,
        chat_messages: List[Dict[str, str]],
        stream: bool = False,
        citation: bool = False,
        with_intent: bool = False,
        index_name: str = None,
        search_web: bool = False,
        return_reference: bool = False,
    ):
        query = RagQuery(
            messages=chat_messages,
            stream=stream,
            citation=citation,
            with_intent=with_intent,
            index_name=index_name,
            search_web=search_web,
            return_reference=return_reference,
        )

        try:
            response = await rag_service.aquery_v1(query)
            if isinstance(response, RagResponse):
                result = {
                    "delta": response.answer,
                    "docs": response.docs,
                }
                yield self._format_rag_response(result)
            else:
                async for r in response:
                    if r.startswith("data: "):
                        chunk = json.loads(r[6:])
                        result = {
                            "delta": chunk["delta"],
                            "docs": chunk.get("docs"),
                            "is_finished": chunk.get("is_finished", False),
                        }
                        yield self._format_rag_response(result)
        except Exception as e:
            raise RagApiError(code=500, msg=str(e))

    async def query_data_analysis(
        self,
        chat_messages: List[Dict[str, str]],
        stream: bool = False,
        return_reference: bool = True,
    ):
        query = RagQuery(
            messages=chat_messages,
            stream=stream,
            return_reference=return_reference,
        )

        try:
            response = await rag_service.aquery_data_analysis_v1(query)
            if isinstance(response, RagResponse):
                result = {
                    "delta": response.answer,
                    "docs": response.docs,
                }
                yield self._format_rag_response(result)
            else:
                async for r in response:
                    if r.startswith("data: "):
                        chunk = json.loads(r[6:])
                        result = {
                            "delta": chunk["delta"],
                            "docs": chunk.get("docs"),
                            "is_finished": chunk.get("is_finished", False),
                        }
                        yield self._format_rag_response(result)
        except Exception as e:
            raise RagApiError(code=500, msg=str(e))

    async def query_llm(
        self,
        chat_messages: List[Dict[str, str]],
        stream: bool = False,
    ):
        query = RagQuery(
            messages=chat_messages,
            stream=stream,
        )

        try:
            response = await rag_service.aquery_llm_v1(query)
            if isinstance(response, RagResponse):
                result = {
                    "delta": response.answer,
                    "docs": response.docs,
                }
                yield self._format_rag_response(result)
            else:
                async for r in response:
                    if r.startswith("data: "):
                        chunk = json.loads(r[6:])
                        result = {
                            "delta": chunk["delta"],
                            "docs": chunk.get("docs"),
                            "is_finished": chunk.get("is_finished", False),
                        }
                        yield self._format_rag_response(result)
        except Exception as e:
            raise RagApiError(code=500, msg=str(e))

    async def query_vector(
        self, chat_messages: List[Dict[str, str]], text: str, index_name: str = None
    ):
        try:
            response = await rag_service.aquery_retrieval(
                RagQuery(
                    messages=chat_messages,
                    question=text,
                    index_name=index_name,
                )
            )

            result = {}
            formatted_text = (
                "<tr><th>Document</th><th>Score</th><th>Text</th><th>Media</tr>\n"
            )
            if len(response.docs) == 0:
                result["delta"] = EMPTY_KNOWLEDGEBASE_MESSAGE.format(query_str=text)
            else:
                for i, doc in enumerate(response.docs):
                    html_content = markdown.markdown(doc.text)
                    file_url = doc.metadata.get("file_url", None)
                    if doc.image_url:
                        media_url = doc.image_url
                    else:
                        media_url = doc.metadata.get("image_info_list", None)
                    if media_url and isinstance(media_url, list):
                        media_url = "<br>".join(
                            [
                                f'<img src="{url.get("image_url", None)}" alt="Image {j + 1}"/>'
                                for j, url in enumerate(media_url)
                            ]
                        )
                    elif media_url:
                        media_url = f"""<img src="{media_url}"/>"""
                    safe_html_content = html.escape(html_content).replace("\n", "<br>")
                    if file_url:
                        safe_html_content = (
                            f"""<a href="{file_url}">{safe_html_content}</a>"""
                        )
                    formatted_text += '<tr style="font-size: 13px;"><td>Doc {}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n'.format(
                        i + 1, doc.score, safe_html_content, media_url
                    )
                formatted_text = (
                    "<table>\n<tbody>\n" + formatted_text + "</tbody>\n</table>"
                )
                result["delta"] = formatted_text
            yield dotdict(result)

        except Exception as error:
            raise RagApiError(code=500, msg=str(error))

    async def add_knowledge(
        self,
        oss_path: str = None,
        input_files: str = None,
        enable_raptor: bool = False,
        enable_multimodal: bool = False,
        index_name: str = None,
    ):
        task_id = uuid.uuid4().hex
        logger.info(
            f"[Upload] Submitting upload data task_id: {task_id} index_name: {index_name} enable_multimodal: {enable_multimodal}"
        )

        if oss_path:
            upload_job = asyncio.create_task(
                run_in_threadpool(
                    rag_service.add_knowledge,
                    task_id=task_id,
                    filter_pattern=None,
                    oss_path=oss_path,
                    from_oss=True,
                    index_name=index_name,
                    enable_raptor=enable_raptor,
                    enable_multimodal=enable_multimodal,
                )
            )
        else:
            tmpdir = f"./localdata/uploaddata/local/{get_ts()}"
            converted_input_file_list = []
            for file in input_files:
                with open(file, "rb") as fi:
                    data = fi.read()
                    file_hash = hashlib.md5(data).hexdigest()
                    tmp_file_dir = os.path.join(
                        tmpdir, f"{COMMON_FILE_PATH_FODER_NAME}/{file_hash}"
                    )
                    os.makedirs(tmp_file_dir, exist_ok=True)
                    save_file = os.path.join(tmp_file_dir, file)

                    with open(save_file, "wb") as f:
                        f.write(data)
                        f.close()
                    converted_input_file_list.append(save_file)

            upload_job = asyncio.create_task(
                run_in_threadpool(
                    rag_service.add_knowledge,
                    task_id=task_id,
                    input_files=converted_input_file_list,
                    filter_pattern=None,
                    index_name=index_name,
                    oss_path=None,
                    enable_raptor=enable_raptor,
                    temp_file_dir=tmpdir,
                    enable_multimodal=enable_multimodal,
                )
            )

        logger.info(
            f"[Upload] Submitted upload data task_id: {task_id} index_name: {index_name} enable_multimodal: {enable_multimodal}"
        )

        result = {"Info": ["StartTime", "EndTime", "Duration(s)", "Status"]}
        start = time.time()
        while True:
            status, detail = rag_service.get_task_status(task_id=task_id)
            duration = time.time() - start
            logger.info(
                f"[Upload] task_id: {task_id}, status: {status}, duration: {duration}"
            )
            result = {
                "status": [status],
                "duration": [duration],
                "detail": [detail],
            }
            yield result

            if status in ["completed", "failed"]:
                break

            await asyncio.sleep(2)

        await upload_job
        logger.info(f"[Upload] Finished task_id: {task_id}")

    def handle_task_result(self, task, task_id):
        try:
            # 尝试获取任务的结果，以捕捉异常
            while True:
                status, _ = rag_service.get_task_status(task_id=task_id)
                if status in ["completed", "failed"]:
                    break
        except Exception as e:
            logger.error(f"Upload job {task_id} failed: {e}")

    async def async_add_knowledge_file(
        self,
        oss_path: str = None,
        input_files: str = None,
        enable_raptor: bool = False,
        enable_multimodal: bool = False,
        index_name: str = None,
    ):
        task_id = uuid.uuid4().hex
        logger.info(
            f"[Upload] Submitting upload data task_id: {task_id} index_name: {index_name} enable_multimodal: {enable_multimodal}"
        )

        if oss_path:
            upload_job = asyncio.create_task(
                run_in_threadpool(
                    rag_service.add_knowledge,
                    task_id=task_id,
                    filter_pattern=None,
                    oss_path=oss_path,
                    from_oss=True,
                    index_name=index_name,
                    enable_raptor=enable_raptor,
                    enable_multimodal=enable_multimodal,
                )
            )
        else:
            upload_job = asyncio.create_task(
                run_in_threadpool(
                    rag_service.add_knowledge,
                    task_id=task_id,
                    input_files=input_files,
                    filter_pattern=None,
                    index_name=index_name,
                    oss_path=None,
                    enable_raptor=enable_raptor,
                    enable_multimodal=enable_multimodal,
                )
            )
        # 为任务添加回调，以处理可能的异常
        upload_job.add_done_callback(lambda t: self.handle_task_result(t, task_id))
        logger.info(
            f"[Upload] Submitted upload data task_id: {task_id} index_name: {index_name} enable_multimodal: {enable_multimodal}, input_files: {input_files}"
        )

        return

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
            await rag_service.aload_db_info()
        except Exception as e:
            logger.exception(f"load db info failed: {e}")
            raise  # 重新抛出异常

    def get_knowledge_state(self, task_id: str):
        status, detail = rag_service.get_task_status(task_id=task_id)
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
            config = rag_service.reload(new_config)
            return
        except Exception as e:
            logger.exception(f"patch config failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"patch config failed. {e}",
            )

    def get_config(self):
        try:
            config = rag_service.get_config()
            rag_config = RagConfig.model_validate(config)
            # 兼容之前的配置
            if rag_config.llm:
                updated_llm = rag_config.llm.copy(update={"vision_support": False})
                rag_config.llms.append(updated_llm)
            if rag_config.multimodal_llm:
                updated_vllm = rag_config.multimodal_llm.copy(
                    update={"vision_support": True}
                )
                rag_config.llms.append(updated_vllm)
            return rag_config

        except Exception as e:
            logger.exception(f"get config failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"get config failed. {e}",
            )

    def list_indexes(self) -> RagIndexMap:
        try:
            return index_manager.list_indexes()
        except Exception as e:
            logger.exception(f"list index failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"list index failed. {e}",
            )

    def add_index(self, index_entry: RagIndexEntry):
        try:
            index_manager.add_index(index_entry=index_entry)
        except Exception as e:
            logger.exception(f"add index {index_entry.index_name} failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"add index {index_entry.index_name} failed. {e}",
            )

    def update_index(self, index_entry: RagIndexEntry):
        try:
            index_manager.update_index(index_entry=index_entry)
        except Exception as e:
            logger.exception(f"update index {index_entry.index_name} failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"update index {index_entry.index_name} failed. {e}",
            )

    def delete_index(self, index_name: str):
        try:
            index_manager.delete_index(index_name=index_name)
        except Exception as e:
            logger.exception(f"delete index {index_name} failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"delete index {index_name} failed. {e}",
            )

    def add_file_to_index(self, index_name: str, file_path: str):
        try:
            index_manager.add_file_to_index(index_name=index_name, file_path=file_path)
        except Exception as e:
            logger.exception(f"Add file {file_path} to_index {index_name} failed: {e}")
            raise RagApiError(
                code=500,
                msg=f"Add file {file_path} to_index {index_name} failed. {e}",
            )

    def delete_file_from_index(self, index_name: str, file_path: str):
        return index_manager.delete_file_from_index(
            index_name=index_name, file_path=file_path
        )

    def delete_dir_from_index(self, index_name: str, dir_path: str):
        return index_manager.delete_dir_from_index(
            index_name=index_name, file_path=dir_path
        )


rag_client = RagLocalClient()
