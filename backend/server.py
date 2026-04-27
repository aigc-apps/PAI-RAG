import json
import os

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.agent_service import (
    AgentService,
    last_user_text,
    openai_chat_chunk,
    openai_chat_completion,
    openai_done_chunk,
)

try:
    import config
except ImportError as e:
    raise RuntimeError('config.py not found. Copy config_template.py to config.py first.') from e


app = FastAPI(title='PAI-RAG OpenAI Compatible Backend')
service = AgentService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, 'SERVER_CORS_ORIGINS', ['*']),
    allow_methods=['*'],
    allow_headers=['*'],
)


def require_auth(authorization: str | None = Header(default=None)):
    expected = getattr(config, 'SERVER_API_KEY', '') or os.environ.get('SERVER_API_KEY', '')
    if not expected:
        return
    if authorization != f'Bearer {expected}':
        raise HTTPException(status_code=401, detail='Unauthorized')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/v1/chat/completions')
async def chat_completions(
    request: Request,
    _: None = Depends(require_auth),
    x_session_id: str | None = Header(default=None),
):
    body = await request.json()
    model = body.get('model') or getattr(config, 'MODEL', 'qwen-plus')
    messages = body.get('messages') or []
    stream = bool(body.get('stream'))
    cwd = body.get('cwd')
    if not last_user_text(messages).strip():
        raise HTTPException(status_code=400, detail='messages must include a non-empty user message')

    sess, text_iter = service.chat_text(x_session_id, messages, cwd=cwd)
    headers = {'X-Session-Id': sess.sid}

    if stream:
        def event_stream():
            for text in text_iter:
                if not text:
                    continue
                chunk = openai_chat_chunk(model, text)
                yield f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
            yield f'data: {json.dumps(openai_done_chunk(model), ensure_ascii=False)}\n\n'
            yield 'data: [DONE]\n\n'

        return StreamingResponse(event_stream(), media_type='text/event-stream', headers=headers)

    content = ''.join(text_iter)
    payload = openai_chat_completion(model, content, sess.sid)
    return JSONResponse(payload, headers=headers)


@app.get('/v1/sessions')
def list_sessions(_: None = Depends(require_auth)):
    return {'object': 'list', 'data': service.list_sessions()}


@app.post('/v1/sessions')
async def create_session(request: Request, _: None = Depends(require_auth)):
    body = await request.json() if request.headers.get('content-length') not in (None, '0') else {}
    sess = service.create_session(cwd=(body or {}).get('cwd'))
    loaded = service.store.load(sess.sid) or {}
    return {
        'session_id': sess.sid,
        'title': loaded.get('title', 'New Chat'),
        'created_at': loaded.get('created_at', ''),
        'updated_at': loaded.get('updated_at', ''),
        'messages': loaded.get('ui_messages', []),
    }


@app.get('/v1/sessions/{session_id}')
def get_session(session_id: str, _: None = Depends(require_auth)):
    loaded = service.store.load(session_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail='Session not found')
    payload = {
        'session_id': session_id,
        'title': loaded.get('title', 'New Chat'),
        'created_at': loaded.get('created_at', ''),
        'updated_at': loaded.get('updated_at', ''),
        'messages': loaded.get('ui_messages', []),
    }
    if getattr(config, 'EXPOSE_SESSION_DEBUG', False):
        payload['llm_history'] = loaded.get('llm_history', [])
        payload['handler_state'] = loaded.get('handler_state')
    return payload


@app.delete('/v1/sessions/{session_id}')
def delete_session(session_id: str, _: None = Depends(require_auth)):
    deleted = service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail='Session not found')
    return {'deleted': True, 'session_id': session_id}


@app.post('/v1/sessions/{session_id}/cancel')
def cancel_session(session_id: str, _: None = Depends(require_auth)):
    cancelled = service.cancel_session(session_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail='Session not found')
    return {'cancelled': True, 'session_id': session_id}
