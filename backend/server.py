import json
import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.agent_service import (
    AgentService,
    ServiceCapacityError,
    SessionBusyError,
    last_user_text,
    openai_chat_chunk,
    openai_chat_completion,
    openai_done_chunk,
    openai_role_chunk,
)
from backend.auth import AuthContext, UserStore, create_token, decode_token
from backend.workspace import WorkspaceViolation
from session_store import SERVER_USER_ID
from tools import WorkspaceViolation as ToolWorkspaceViolation

try:
    import config
except ImportError as e:
    raise RuntimeError('config.py not found. Copy config_template.py to config.py first.') from e


app = FastAPI(title='PAI-RAG OpenAI Compatible Backend')
service = AgentService()
RUNNER_BACKEND = (os.environ.get('RUNNER_BACKEND') or getattr(config, 'RUNNER_BACKEND', 'thread') or 'thread').lower()
celery_service = None
if RUNNER_BACKEND == 'celery':
    from backend.celery_runner import CeleryRunService

    celery_service = CeleryRunService(service.store, service.workspace_manager)
user_store = UserStore(service.store.db_path)
AUTH_SECRET = getattr(config, 'AUTH_SECRET', '') or os.environ.get('AUTH_SECRET', '')
if not AUTH_SECRET:
    AUTH_SECRET = secrets.token_urlsafe(32)
    print('[Warn] AUTH_SECRET is not configured. Using a temporary development secret; tokens will expire on restart.')
AUTH_TOKEN_TTL_SECONDS = int(getattr(config, 'AUTH_TOKEN_TTL_SECONDS', 7 * 24 * 60 * 60))

app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, 'SERVER_CORS_ORIGINS', ['*']),
    allow_methods=['*'],
    allow_headers=['*'],
)


def auth_payload(user):
    return {
        'access_token': create_token(user, AUTH_SECRET, AUTH_TOKEN_TTL_SECONDS),
        'token_type': 'bearer',
        'user': user,
    }


def require_auth(authorization: str | None = Header(default=None)):
    expected = getattr(config, 'SERVER_API_KEY', '') or os.environ.get('SERVER_API_KEY', '')
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Unauthorized')
    token = authorization.removeprefix('Bearer ').strip()
    if expected and token == expected:
        return AuthContext(user_id=SERVER_USER_ID, username='server', is_service=True)
    try:
        claims = decode_token(token, AUTH_SECRET)
    except Exception as e:
        raise HTTPException(status_code=401, detail='Unauthorized') from e
    user = user_store.get_by_id(claims.get('user_id'))
    if not user:
        raise HTTPException(status_code=401, detail='Unauthorized')
    return AuthContext(user_id=user['user_id'], username=user['username'])


def backend_error(exc):
    if isinstance(exc, SessionBusyError):
        return HTTPException(
            status_code=409,
            detail={'code': 'session_busy', 'message': str(exc), 'status': exc.status},
        )
    if isinstance(exc, ServiceCapacityError):
        return HTTPException(
            status_code=429,
            detail={'code': 'capacity_exceeded', 'message': str(exc), 'scope': exc.scope, 'limit': exc.limit},
        )
    if isinstance(exc, (WorkspaceViolation, ToolWorkspaceViolation)):
        return HTTPException(status_code=400, detail={'code': 'workspace_violation', 'message': str(exc)})
    return exc


def stream_headers(session_id, run_id=''):
    return {
        'X-Session-Id': session_id,
        'X-Run-Id': run_id or '',
        'Cache-Control': 'no-cache, no-transform',
        'X-Accel-Buffering': 'no',
    }


def start_celery_run(session_id, user_id, text, mode='events', cwd=None):
    if celery_service is None:
        raise HTTPException(status_code=500, detail='Celery runner is not enabled')
    try:
        result = celery_service.start_or_answer(session_id, user_id, text, mode=mode, cwd=cwd)
    except (SessionBusyError, ServiceCapacityError, WorkspaceViolation, ToolWorkspaceViolation) as e:
        raise backend_error(e) from e
    if result is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return result


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/v1/auth/register')
async def register(request: Request):
    body = await request.json()
    try:
        user = user_store.create_user(body.get('username'), body.get('password'))
    except ValueError as e:
        detail = str(e)
        status_code = 409 if 'already exists' in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from e
    return auth_payload(user)


@app.post('/v1/auth/login')
async def login(request: Request):
    body = await request.json()
    user = user_store.authenticate(body.get('username'), body.get('password'))
    if not user:
        raise HTTPException(status_code=401, detail='Invalid username or password')
    return auth_payload(user)


@app.get('/v1/auth/me')
def me(auth: AuthContext = Depends(require_auth)):
    return {
        'user_id': auth.user_id,
        'username': auth.username,
        'is_service': auth.is_service,
    }


@app.post('/v1/chat/completions')
async def chat_completions(
    request: Request,
    auth: AuthContext = Depends(require_auth),
    x_session_id: str | None = Header(default=None),
):
    body = await request.json()
    model = body.get('model') or getattr(config, 'MODEL', 'qwen-plus')
    messages = body.get('messages') or []
    stream = bool(body.get('stream'))
    cwd = body.get('cwd')
    if not last_user_text(messages).strip():
        raise HTTPException(status_code=400, detail='messages must include a non-empty user message')

    if celery_service is not None:
        if x_session_id:
            session_id = x_session_id
        else:
            try:
                sess = service.create_session(user_id=auth.user_id, cwd=cwd)
            except WorkspaceViolation as e:
                raise backend_error(e) from e
            session_id = sess.sid
        run = start_celery_run(session_id, auth.user_id, last_user_text(messages), mode='text', cwd=cwd)
        headers = stream_headers(run.session_id, run.run_id)
        response_session_id = run.session_id

        def text_events():
            for update in celery_service.iter_events(run.run_id, run.stream_from):
                if update.get('sessionUpdate') == 'agent_message_chunk':
                    yield ((update.get('content') or {}).get('text') or '')

        text_iter = text_events()
    else:
        try:
            sess, text_iter = service.chat_text(x_session_id, messages, user_id=auth.user_id, cwd=cwd)
        except (SessionBusyError, ServiceCapacityError, WorkspaceViolation, ToolWorkspaceViolation) as e:
            raise backend_error(e) from e
        if sess is None:
            raise HTTPException(status_code=404, detail='Session not found')
        headers = stream_headers(sess.sid, getattr(sess, 'active_run_id', '') or '')
        response_session_id = sess.sid

    if stream:
        def event_stream():
            yield f'data: {json.dumps(openai_role_chunk(model), ensure_ascii=False)}\n\n'
            for text in text_iter:
                if not text:
                    continue
                chunk = openai_chat_chunk(model, text)
                yield f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
            yield f'data: {json.dumps(openai_done_chunk(model), ensure_ascii=False)}\n\n'
            yield 'data: [DONE]\n\n'

        return StreamingResponse(event_stream(), media_type='text/event-stream', headers=headers)

    content = ''.join(text_iter)
    payload = openai_chat_completion(model, content, response_session_id)
    return JSONResponse(payload, headers=headers)


@app.post('/v1/agent/sessions/{session_id}/prompt')
async def agent_prompt(
    session_id: str,
    request: Request,
    auth: AuthContext = Depends(require_auth),
):
    body = await request.json()
    text = body.get('message') or body.get('text') or last_user_text(body.get('messages') or [])
    cwd = body.get('cwd')
    if not str(text or '').strip():
        raise HTTPException(status_code=400, detail='message must be non-empty')

    if celery_service is not None:
        run = start_celery_run(session_id, auth.user_id, str(text), mode='events', cwd=cwd)
        event_iter = celery_service.iter_events(run.run_id, run.stream_from)
        headers = stream_headers(run.session_id, run.run_id)
        response_session_id = run.session_id
    else:
        try:
            sess, event_iter = service.chat_events(session_id, str(text), user_id=auth.user_id, cwd=cwd)
        except (SessionBusyError, ServiceCapacityError, WorkspaceViolation, ToolWorkspaceViolation) as e:
            raise backend_error(e) from e
        if sess is None:
            raise HTTPException(status_code=404, detail='Session not found')
        headers = stream_headers(sess.sid, getattr(sess, 'active_run_id', '') or '')
        response_session_id = sess.sid

    def event_stream():
        for update in event_iter:
            payload = {'sessionId': response_session_id, 'update': update}
            yield f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'

    return StreamingResponse(event_stream(), media_type='text/event-stream', headers=headers)


@app.get('/v1/sessions')
def list_sessions(auth: AuthContext = Depends(require_auth)):
    return {'object': 'list', 'data': service.list_sessions(user_id=auth.user_id)}


@app.post('/v1/sessions')
async def create_session(request: Request, auth: AuthContext = Depends(require_auth)):
    body = await request.json() if request.headers.get('content-length') not in (None, '0') else {}
    try:
        sess = service.create_session(user_id=auth.user_id, cwd=(body or {}).get('cwd'))
    except WorkspaceViolation as e:
        raise backend_error(e) from e
    loaded = service.store.load(sess.sid, user_id=auth.user_id) or {}
    return {
        'session_id': sess.sid,
        'title': loaded.get('title', 'New Task'),
        'created_at': loaded.get('created_at', ''),
        'updated_at': loaded.get('updated_at', ''),
        'status': loaded.get('status', 'idle'),
        'active_run_id': loaded.get('active_run_id', ''),
        'messages': loaded.get('ui_messages', []),
    }


@app.get('/v1/sessions/{session_id}')
def get_session(session_id: str, auth: AuthContext = Depends(require_auth)):
    loaded = service.store.load(session_id, user_id=auth.user_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail='Session not found')
    payload = {
        'session_id': session_id,
        'title': loaded.get('title', 'New Task'),
        'created_at': loaded.get('created_at', ''),
        'updated_at': loaded.get('updated_at', ''),
        'status': loaded.get('status', 'idle'),
        'active_run_id': loaded.get('active_run_id', ''),
        'messages': loaded.get('ui_messages', []),
    }
    if getattr(config, 'EXPOSE_SESSION_DEBUG', False):
        payload['llm_history'] = loaded.get('llm_history', [])
        payload['handler_state'] = loaded.get('handler_state')
    return payload


@app.delete('/v1/sessions/{session_id}')
def delete_session(session_id: str, auth: AuthContext = Depends(require_auth)):
    deleted = service.delete_session(session_id, user_id=auth.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail='Session not found')
    return {'deleted': True, 'session_id': session_id}


@app.post('/v1/sessions/{session_id}/cancel')
def cancel_session(session_id: str, auth: AuthContext = Depends(require_auth)):
    if celery_service is not None:
        cancelled = celery_service.cancel_session(session_id, user_id=auth.user_id)
    else:
        cancelled = service.cancel_session(session_id, user_id=auth.user_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail='Session not found')
    return {'cancelled': True, 'session_id': session_id}
