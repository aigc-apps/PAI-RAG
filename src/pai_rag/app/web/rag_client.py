import json

from typing import Any
import requests

cache_config = None


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
    def load_knowledge_url(self):
        return f"{self.endpoint}service/knowledge"

    def query(self, text: str, session_id: str = None):
        q = dict(question=text)
        r = requests.post(self.query_url, headers={"X-Session-ID": session_id}, json=q)
        r.raise_for_status()
        session_id = r.headers["x-session-id"]
        response = dotdict(json.loads(r.text))
        response.session_id = session_id

        return response

    def query_llm(
        self,
        text: str,
        session_id: str = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        eas_llm_top_k: float = 30,
    ):
        q = dict(question=text, topp=top_p, topk=eas_llm_top_k, temperature=temperature)

        r = requests.post(self.llm_url, headers={"X-Session-ID": session_id}, json=q)
        r.raise_for_status()
        session_id = r.headers["x-session-id"]
        response = dotdict(json.loads(r.text))
        response.session_id = session_id

        return response

    def query_vector(self, text: str):
        q = dict(question=text)
        r = requests.post(self.retrieval_url, json=q)
        r.raise_for_status()
        session_id = r.headers["x-session-id"]
        response = dotdict(json.loads(r.text))
        response.session_id = session_id

        return response

    def add_knowledge(self, file_dir: str, enable_qa_extraction: bool):
        q = dict(file_path=file_dir, enable_qa_extraction=enable_qa_extraction)
        r = requests.post(self.load_knowledge_url, json=q)
        r.raise_for_status()
        return

    def reload_config(self, config: Any):
        global cache_config

        if cache_config == config:
            return

        r = requests.patch(self.config_url, json=config)
        r.raise_for_status()
        print(r.text)
        cache_config = config

        return


rag_client = RagWebClient()
