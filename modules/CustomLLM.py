from langchain.llms.base import LLM
import time
import logging
import requests
from typing import Optional, List, Mapping, Any
from loguru import logger

class CustomLLM(LLM):
    
    # # 模型服务url
    url = ""
    token = ""
    history = []
    top_k = ""
    top_p = ""
    temperature = ""
        
    @property
    def _llm_type(self) -> str:
        return "custom llm"
 
    def _construct_query(self, prompt: str) -> str:
        """构造请求体
        """
        query = prompt.encode('utf8')
        return query
 
    @classmethod
    def _post(cls, url: str,
        query: str, token: str) -> Any:
        """POST请求
        """
        _headers = {
            "Authorization": token,
            'Accept': "*/*",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        with requests.session() as sess:
            resp = sess.post(url, 
                json=query, 
                headers=_headers, 
                timeout=10000)
        return resp
 
    def _call(self, prompt: str,
        stop: Optional[List[str]] = None) -> str:
        """_call
        """        
        query_json = {
            "prompt": str(prompt),
            "history": self.history,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature
        }
        
        # post
        logger.info('query_json',query_json)
        _headers = {
            "Authorization": self.token,
            'Accept': "*/*",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        resp = requests.post(self.url, 
            json=query_json, 
            headers=_headers, 
            timeout=10000)
        if resp.status_code == 200:
            resp_json = resp.json()
            predictions = resp_json["response"]
            return predictions
        else:
            return resp.text
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.
        """
        _param_dict = {
            "url": self.url,
            "token": self.token,
            "history":self.history
        }
        return _param_dict