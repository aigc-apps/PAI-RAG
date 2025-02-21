import aiohttp
from fastapi import Request, Response
from typing import Awaitable, Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

def clean_headers(headers:dict,keys):
    for k in keys:
        headers.pop(k, None)
        headers.pop(str.lower(k),None)
    return headers

class FileBrowserProxy:
    def __init__(self, app):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total= 30*60,
                connect= 30*60,
            )
        )
        self.app = app
        self.host = 'localhost' 
        self.port = 8012
    
    async def inner_send(self, message: Message, send: Send, status_code: list[int]):
        """
        Summary
        -------
        a function to log the requests

        Parameters
        ----------
        message (Message) : the message
        send (Send) : the send function
        status_code (list[int]) : the status code
        """
        if message['type'] == 'http.response.start':
            status_code[0] = message['status']

        await send(message)


    def inner_send_factory(self, send: Send, status_code: list[int]) -> Callable[[Message], Awaitable[None]]:
        """
        Summary
        -------
        a factory to create the inner send function

        Parameters
        ----------
        send (Send) : the send function
        status_code (list[int]) : the status code

        Returns
        -------
        inner_send (Callable[[Message], Awaitable[None]]) : a custom send function
        """
        return lambda message: self.inner_send(message, send, status_code)
    async def __call__(self, scope, receive, send):
        path = scope['path']
        try:
            if '/filebrowser' in path:
                req = Request(scope,receive,send)
                url = req.url.replace(hostname=self.host, port=self.port)
                async def sender_data(req:Request):
                    async for chunk in req.stream():
                        yield chunk
                async with self.session.request(req.method, str(url),headers=clean_headers(dict(req.headers),["Transfer-Encoding"]),params=str(req.path_params), data=sender_data(req), allow_redirects=False) as resp:
                    content = await resp.content.read()
                    r = Response(content=content,headers=clean_headers(dict(resp.headers),["Content-Encoding","Content-Length"]), status_code=resp.status)
                    await r(scope,receive,send)
                    return 
            else:
                return await self.app(scope, receive, self.inner_send_factory(send, [500]))
        except Exception as e:
            print(f'{path} failed with exception {e}')
            return
