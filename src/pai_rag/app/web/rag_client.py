import json
from typing import Any
import requests
import html
import markdown
import httpx
import os
import mimetypes
from pai_rag.app.web.view_model import ViewModel


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class RagWebClient:
    def __init__(self):
        self.endpoint = "http://127.0.0.1:8000/"  # default link

    def set_endpoint(self, endpoint: str):
        self.endpoint = endpoint

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
        return f"{self.endpoint}service/upload_local_data"

    @property
    def get_load_state_url(self):
        return f"{self.endpoint}service/get_upload_state"

    def query(self, text: str, session_id: str = None):
        q = dict(question=text, session_id=session_id)
        r = requests.post(self.query_url, json=q)
        r.raise_for_status()
        response = dotdict(json.loads(r.text))
        return response

    def query_llm(
        self,
        text: str,
        session_id: str = None,
        temperature: float = 0.1,
    ):
        q = dict(
            question=text,
            temperature=temperature,
            session_id=session_id,
        )

        r = requests.post(self.llm_url, json=q)
        r.raise_for_status()
        response = dotdict(json.loads(r.text))

        return response

    def query_vector(self, text: str):
        q = dict(question=text)
        r = requests.post(self.retrieval_url, json=q)
        r.raise_for_status()
        response = dotdict(json.loads(r.text))
        formatted_text = "<tr><th>Document</th><th>Score</th><th>Text</th></tr>\n"
        for i, doc in enumerate(response["docs"]):
            html_content = markdown.markdown(doc["text"])
            safe_html_content = html.escape(html_content).replace("\n", "<br>")
            formatted_text += '<tr style="font-size: 13px;"><td>Doc {}</td><td>{}</td><td>{}</td></tr>\n'.format(
                i + 1, doc["score"], safe_html_content
            )
        formatted_text = "<table>\n<tbody>\n" + formatted_text + "</tbody>\n</table>"
        response["answer"] = formatted_text
        return response

    def add_knowledge(self, file_name: str, enable_qa_extraction: bool):
        with open(file_name, "rb") as file_obj:
            mimetype = mimetypes.guess_type(file_name)[0]
            # maybe we can upload multiple files in the future
            files = {"file": (os.path.basename(file_name), file_obj, mimetype)}
            print(files)
            # headers = {"content-type": "multipart/form-data"}

            r = requests.post(
                self.load_data_url,
                files=files,
                # headers=headers,
            )
        r.raise_for_status()
        response = dotdict(json.loads(r.text))
        return response

    async def get_knowledge_state(self, task_id: str):
        async with httpx.AsyncClient() as client:
            r = await client.get(self.get_load_state_url, params={"task_id": task_id})
            r.raise_for_status()
            response = dotdict(json.loads(r.text))
            return response

    def patch_config(self, update_dict: Any):
        config = self.get_config()
        view_model: ViewModel = ViewModel.from_app_config(config)
        view_model.update(update_dict)
        new_config = view_model.to_app_config()

        r = requests.patch(self.config_url, json=new_config)
        r.raise_for_status()
        return

    def get_config(self):
        r = requests.get(self.config_url)
        r.raise_for_status()
        response = dotdict(json.loads(r.text))
        print(response)
        return response


rag_client = RagWebClient()
