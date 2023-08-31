from langchain.llms.base import LLM
import time
import logging
import requests
from typing import Optional, List, Mapping, Any
 

class ChatGLM(LLM):
    
    # # 模型服务url
    url = ""
    token = ""
        
    @property
    def _llm_type(self) -> str:
        return "chatglm"
 
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
                data=query, 
                headers=_headers, 
                timeout=10000)
        return resp
 
    
    def _call(self, prompt: str, 
        stop: Optional[List[str]] = None) -> str:
        """_call
        """
        # construct query
        query = self._construct_query(prompt=prompt)
        # post
        resp = self._post(url=self.url,
            query=query, token = self.token)
        if resp.status_code == 200:
            resp_json = resp.json()
            predictions = resp_json["response"]
            return predictions
        else:
            return "There may occur some errors." 
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.
        """
        _param_dict = {
            "url": self.url,
            "token": self.token
        }
        return _param_dict

