import argparse
import json
from typing import Iterable, List

import requests

def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    delimiter = b'\0'
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=delimiter):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            
            output = data["response"]
            tokens = data["tokens"]
            yield output, tokens

def post_http_request():
    host = "http://localhost:8000/chat/llm"
    headers = {
        "Content-Type": "application/json"
    }
    
    pload = {
        "question": "杭州怎么样？", 
        "vector_topk":10,
        "score_threshold":700,
        "use_chat_stream": True
    }

    response = requests.post(host, headers=headers,
                             json=pload)
    return response

response = post_http_request()
res = ""
for h, tokens in get_streaming_response(response):
    res += h
    print(
        f" --- stream line: {res} \n --- tokens: {tokens}", flush=True)