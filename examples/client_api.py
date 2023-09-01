import requests

_URL = 'http://127.0.0.1:8098'

def test_post_api_config():
    # CMD: curl -X 'POST' 'http://127.0.0.1:8098/config' -H 'accept: application/json'  -H 'Content-Type: multipart/form-data'  -H 'Authorization: ==' -F 'file=@config_holo.json'
    
    url = _URL + '/config'
    headers = {
        # 'Authorization': '==',
    }
    files = {'file': (open('config_holo.json', 'rb'))}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code != 200:
        raise ValueError(f'Error post to {url}, code: {response.status_code}')
    ans = response.json()
    return ans['response']

def test_post_api_uploafile():
    # CMD: curl -X 'POST' 'http://127.0.0.1:8098/uploadfile/' -H 'accept: application/json'  -H 'Content-Type: multipart/form-data'  -F 'file=@docs/PAI.txt;type=text/plain'
    
    url = _URL + '/uploadfile'
    headers = {
        # 'Authorization': '==',
    }
    files = {'file': (open('docs/PAI.txt', 'rb'))}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code != 200:
        raise ValueError(f'Error post to {url}, code: {response.status_code}')
    ans = response.json()
    return ans['response']

def test_post_api_chat_vectorstore():
    # CMD: curl -X 'POST'  'http://127.0.0.1:8098/chat/vectorstore'  -H 'accept: application/json'  -H 'Content-Type: application/json'  -d '{"question": "什么是机器学习PAI?"}'
    
    url = _URL + '/chat/vectorstore'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        'question': '什么是机器学习PAI?'
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f'Error post to {url}, code: {response.status_code}')
    ans = response.json()
    return ans['response']


def test_post_api_chat_llm():
    # CMD: curl -X 'POST'  'http://127.0.0.1:8098/chat/llm'  -H 'accept: application/json'  -H 'Content-Type: application/json'  -d '{"question": "什么是机器学习PAI?"}'
    
    url = _URL + '/chat/llm'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        'question': '什么是机器学习PAI?'
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f'Error post to {url}, code: {response.status_code}')
    ans = response.json()
    return ans['response']


def test_post_api_chat_langchain():
    # CMD: curl -X 'POST'  'http://127.0.0.1:8098/chat/langchain'  -H 'accept: application/json'  -H 'Content-Type: application/json'  -d '{"question": "什么是机器学习PAI?"}'
    
    url = _URL + '/chat/langchain'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        'question': '什么是机器学习PAI?'
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f'Error post to {url}, code: {response.status_code}')
    ans = response.json()
    return ans['response']

print(test_post_api_chat_vectorstore())