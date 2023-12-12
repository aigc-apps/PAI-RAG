import argparse
import json
from typing import Iterable, List

import requests

class EasLlmClient:
    def __init__(
        self,
        host: str,
        authorization: str,
        max_new_tokens: int = 2048,
        langchain: bool = False,
        no_template: bool = False
    ) -> None:
        self. host = host
        self.authorization = authorization
        self.max_new_tokens = max_new_tokens
        self.langchain = langchain
        self.no_template = no_template

            
    def post_http_request(self,prompt,system_prompt,history,temperature,top_k,top_p,use_stream_chat=None,**kwargs) -> requests.Response:
        headers = {
            "User-Agent": "Test Client",
            "Authorization": f"{self.authorization}"
        }
        _temperature=float(temperature) if (temperature is not None) else float(0.7)
        _top_k=int(top_k) if (top_k is not None) else int(30)
        _top_p=float(top_p) if (top_p is not None) else float(0.8)
            
        pload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "top_k": _top_k,
            "top_p": _top_p,
            "temperature": _temperature,
            "max_new_tokens": self.max_new_tokens,
            "use_stream_chat": use_stream_chat,
            "history": history,
            "stop": ["."],
            "no_template": self.no_template,
        }

        for k in kwargs:
            pload[k] = kwargs[k]

        if self.langchain:
            print("call for langchain api...")
            pload["langchain"] = self.langchain
        response = requests.post(self.host, headers=headers,
                                json=pload, stream=use_stream_chat)
        return response

    def get_streaming_response(self, response: requests.Response) -> Iterable[List[str]]:
        delimiter = b'\0'  # selected in [b'\n', b'\0']
        print(f"response: {response}")
        for chunk in response.iter_lines(chunk_size=8192,
                                        decode_unicode=False,
                                        delimiter=delimiter):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                
                output = data["response"]
                history = data["history"]
                if "usage" in data:
                    print(data["usage"])

                yield output, history


    def get_response(self, response: requests.Response) -> List[str]:
        data = json.loads(response.content)
        print(f"response: {response}")
        
        output = data["response"]
        history = data["history"]
        if "usage" in data:
            print(data["usage"])
        return output, history