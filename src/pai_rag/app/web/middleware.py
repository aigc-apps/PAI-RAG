import aiohttp
from fastapi import Request, Response

def clean_headers(headers:dict,keys):
    for k in keys:
        headers.pop(k, None)
        headers.pop(str.lower(k),None)
    return headers

class FileBrowserProxy:
    def __init__(self, app, port=8012):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total= 30*60,
                connect= 30*60,
            )
        )
        self.app = app
        self.host = 'localhost' 
        self.port = port
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
                await self.app(scope, receive, send)
                return
        except Exception as e:
            print(f'{path} failed with exception {e}')
            return