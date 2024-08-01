import json
from typing import Any
import requests
import html
import markdown
import httpx
import os
import re
import mimetypes
from http import HTTPStatus
from pai_rag.app.web.view_model import ViewModel
from pai_rag.app.web.ui_constants import EMPTY_KNOWLEDGEBASE_MESSAGE

DEFAULT_CLIENT_TIME_OUT = 60


class RagApiError(Exception):
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class RagWebClient:
    def __init__(self):
        self.endpoint = "http://127.0.0.1:8000/"  # default link

    def set_endpoint(self, endpoint: str):
        self.endpoint = endpoint if endpoint.endswith("/") else f"{endpoint}/"

    @property
    def query_url(self):
        return f"{self.endpoint}service/query"

    @property
    def llm_url(self):
        return f"{self.endpoint}service/query/llm"

    @property
    def retrieval_url(self):
        return f"{self.endpoint}service/query/retrieval"

    @property
    def config_url(self):
        return f"{self.endpoint}service/config"

    @property
    def load_data_url(self):
        return f"{self.endpoint}service/upload_data"

    @property
    def get_load_state_url(self):
        return f"{self.endpoint}service/get_upload_state"

    @property
    def get_evaluate_generate_url(self):
        return f"{self.endpoint}service/evaluate/generate"

    @property
    def get_evaluate_retrieval_url(self):
        return f"{self.endpoint}service/evaluate/retrieval"

    @property
    def get_evaluate_response_url(self):
        return f"{self.endpoint}service/evaluate/response"

    def _format_rag_response(
        self, question, response, session_id: str = None, stream: bool = False
    ):
        if stream:
            text = response["delta"]
        else:
            text = response["answer"]

        docs = response.get("docs", [])
        is_finished = response.get("is_finished", True)

        referenced_docs = ""
        images = ""

        if is_finished and len(docs) == 0 and not text:
            response["result"] = EMPTY_KNOWLEDGEBASE_MESSAGE.format(query_str=question)
            return response
        elif is_finished:
            for i, doc in enumerate(docs):
                filename = doc["metadata"].get("file_name", None)
                if filename:
                    formatted_file_name = re.sub("^[0-9a-z]{32}_", "", filename)
                    referenced_docs += (
                        f'[{i+1}]: {formatted_file_name}   Score:{doc["score"]} \n'
                    )
                image_url = doc["metadata"].get("image_url", None)
                if image_url:
                    images += f"""<img src="{image_url}"/>"""

        formatted_answer = ""
        if session_id:
            new_query = response["new_query"]
            formatted_answer += f"**Query Transformation**: {new_query} \n\n"
        formatted_answer += f"**Answer**: {text} \n\n"
        if images:
            formatted_answer += f"{images} \n\n"
        if referenced_docs:
            formatted_answer += f"**Reference**:\n {referenced_docs}"

        response["result"] = formatted_answer
        return response

    def query(self, text: str, session_id: str = None, stream: bool = False):
        q = dict(question=text, session_id=session_id, stream=stream)
        r = requests.post(self.query_url, json=q, stream=True)
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=r.text)
        if not stream:
            response = dotdict(json.loads(r.text))
            yield self._format_rag_response(
                text, response, session_id=session_id, stream=stream
            )
        else:
            full_content = ""
            for chunk in r.iter_lines(chunk_size=8192, decode_unicode=True):
                chunk_response = dotdict(json.loads(chunk))
                full_content += chunk_response.delta
                chunk_response.delta = full_content
                yield self._format_rag_response(
                    text, chunk_response, session_id=session_id, stream=stream
                )

    def query_llm(
        self,
        text: str,
        session_id: str = None,
        temperature: float = 0.1,
        stream: bool = False,
    ):
        q = dict(
            question=text,
            temperature=temperature,
            session_id=session_id,
            stream=stream,
        )

        r = requests.post(
            self.llm_url, json=q, stream=True, timeout=DEFAULT_CLIENT_TIME_OUT
        )
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=r.text)
        if not stream:
            response = dotdict(json.loads(r.text))
            yield self._format_rag_response(
                text, response, session_id=session_id, stream=stream
            )
        else:
            full_content = ""
            for chunk in r.iter_lines(chunk_size=8192, decode_unicode=True):
                chunk_response = dotdict(json.loads(chunk))
                full_content += chunk_response.delta
                chunk_response.delta = full_content
                yield self._format_rag_response(
                    text, chunk_response, session_id=session_id, stream=stream
                )

    def query_vector(self, text: str):
        q = dict(question=text)
        r = requests.post(self.retrieval_url, json=q, timeout=DEFAULT_CLIENT_TIME_OUT)
        response = dotdict(json.loads(r.text))
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)

        formatted_text = (
            "<tr><th>Document</th><th>Score</th><th>Text</th><th>Media</tr>\n"
        )
        if len(response["docs"]) == 0:
            response["result"] = EMPTY_KNOWLEDGEBASE_MESSAGE.format(query_str=text)
        else:
            for i, doc in enumerate(response["docs"]):
                html_content = markdown.markdown(doc["text"])
                media_url = doc.get("metadata", {}).get("image_url", None)
                if media_url:
                    media_url = f"""<img src="{media_url}"/>"""
                safe_html_content = html.escape(html_content).replace("\n", "<br>")
                formatted_text += '<tr style="font-size: 13px;"><td>Doc {}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n'.format(
                    i + 1, doc["score"], safe_html_content, media_url
                )
            formatted_text = (
                "<table>\n<tbody>\n" + formatted_text + "</tbody>\n</table>"
            )
            response["result"] = formatted_text
        yield response

    def add_knowledge(
        self,
        input_files: str,
        enable_qa_extraction: bool,
        enable_raptor: bool,
        enable_ocr: bool,
        enable_table_summary: bool,
    ):
        files = []
        file_obj_list = []
        for file_name in input_files:
            file_obj = open(file_name, "rb")
            mimetype = mimetypes.guess_type(file_name)[0]
            files.append(("files", (os.path.basename(file_name), file_obj, mimetype)))
            file_obj_list.append(file_obj)
        para = {
            "enable_raptor": enable_raptor,
            "enable_ocr": enable_ocr,
            "enable_table_summary": enable_table_summary,
        }
        try:
            r = requests.post(
                self.load_data_url,
                files=files,
                data=para,
                timeout=DEFAULT_CLIENT_TIME_OUT,
            )
            response = dotdict(json.loads(r.text))
            if r.status_code != HTTPStatus.OK:
                raise RagApiError(code=r.status_code, msg=response.message)
        finally:
            for file_obj in file_obj_list:
                file_obj.close()

        response = dotdict(json.loads(r.text))
        return response

    async def get_knowledge_state(self, task_id: str):
        async with httpx.AsyncClient(timeout=DEFAULT_CLIENT_TIME_OUT) as client:
            r = await client.get(self.get_load_state_url, params={"task_id": task_id})
            response = dotdict(json.loads(r.text))
            if r.status_code != HTTPStatus.OK:
                raise RagApiError(code=r.status_code, msg=response.message)
            return response

    def patch_config(self, update_dict: Any):
        config = self.get_config()
        view_model: ViewModel = ViewModel.from_app_config(config)
        view_model.update(update_dict)
        new_config = view_model.to_app_config()

        r = requests.patch(
            self.config_url, json=new_config, timeout=DEFAULT_CLIENT_TIME_OUT
        )
        response = dotdict(json.loads(r.text))
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)

    def get_config(self):
        r = requests.get(self.config_url, timeout=DEFAULT_CLIENT_TIME_OUT)
        response = dotdict(json.loads(r.text))
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)
        return response

    def evaluate_for_generate_qa(self, overwrite):
        r = requests.post(
            self.get_evaluate_generate_url,
            params={"overwrite": overwrite},
            timeout=DEFAULT_CLIENT_TIME_OUT,
        )
        response = dotdict(json.loads(r.text))
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)
        return response

    def evaluate_for_retrieval_stage(self):
        r = requests.post(
            self.get_evaluate_retrieval_url, timeout=DEFAULT_CLIENT_TIME_OUT
        )
        response = dotdict(json.loads(r.text))

        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)

        return response

    def evaluate_for_response_stage(self):
        r = requests.post(
            self.get_evaluate_response_url, timeout=DEFAULT_CLIENT_TIME_OUT
        )
        response = dotdict(json.loads(r.text))
        if r.status_code != HTTPStatus.OK:
            raise RagApiError(code=r.status_code, msg=response.message)
        print("evaluate_for_response_stage response", response)


rag_client = RagWebClient()
