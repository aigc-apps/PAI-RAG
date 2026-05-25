import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.agent_service import (
    AgentService,
    NoRegeneratableAnswerError,
    ServiceCapacityError,
    SessionBusyError,
    last_user_text,
    long_term_memory_enabled,
)
from backend.background_review import schedule_background_memory_review
from backend.agent_service import handler_memory_scope
from backend.session_archive import archive_session_record
from backend.skills_inventory import skills_inventory
from backend.workspace import WorkspaceViolation
from backend.aliyun_credentials import (
    AliyunConfigError,
    AliyunCredentialError,
    AliyunCredentials,
    AliyunProfileLease,
    pop_aliyun_credentials,
    write_temporary_profile,
)
from session_store import SERVER_USER_ID
from tools import WorkspaceViolation as ToolWorkspaceViolation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import provider_pool
import settings as config
import runtime_config


app = FastAPI(title='PAI-RAG OpenAI Compatible Backend')
service = AgentService()
RUNNER_BACKEND = 'sdk'  # reported by /v1/health/detailed; the SDK runner is the only runtime since Phase 5
SESSION_ARCHIVE_CREATED_AFTER = datetime.now()
# Project never configured the root logger, so every ``logger.info`` from
# ``backend.*`` modules used to get dropped (root default = WARNING). Wire
# a basic handler with a level overridable via env so per-key-acquisition
# audit + warmup + memory-review lines actually reach the worker log.
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger(__name__)

# Lazy SDK-runtime stores: share the same SQLite file as ``service.store``
# so writes from the SDK runner land in the same database as session/response rows.
_sdk_run_state_store = None
_sdk_audit_store = None      # synchronous SQLite writer (used as the consumer sink + fallback)
_sdk_audit_publisher = None  # what the runner calls; wraps Redis when available


_sdk_sweeper_started = False
_sdk_sweeper_lock = threading.Lock()
_sdk_consumer = None
_sdk_consumer_lock = threading.Lock()


def _sdk_stores():
    """Return ``(run_state_store, audit_publisher)``. Constructed on first use.

    The publisher is a :class:`RedisAuditPublisher` when Redis is reachable
    (events go to ``audit:{audit_log_id}`` and an :class:`AuditConsumer`
    daemon drains them into SQLite). When Redis is down, the synchronous
    :class:`AuditStore` is returned directly so audit rows are never lost.

    Also starts the background sweeper that GC's expired ``RunState`` rows
    every ``RUN_STATE_GC_INTERVAL_SECONDS`` and purges audit events older
    than ``AUDIT_RETENTION_DAYS``. Both side effects are idempotent.
    """
    global _sdk_run_state_store, _sdk_audit_store, _sdk_audit_publisher
    global _sdk_sweeper_started, _sdk_consumer
    if _sdk_run_state_store is None:
        from backend.agents_sdk.run_state_store import RunStateStore
        from backend.audit.store import AuditStore
        _sdk_run_state_store = RunStateStore(connect=service.store._connect)
        _sdk_audit_store = AuditStore(connect=service.store._connect)
    if _sdk_audit_publisher is None:
        _sdk_audit_publisher = _build_audit_publisher(_sdk_audit_store)
    if not _sdk_sweeper_started:
        with _sdk_sweeper_lock:
            if not _sdk_sweeper_started:
                _start_sdk_sweeper(_sdk_run_state_store, _sdk_audit_store)
                _sdk_sweeper_started = True
    if _sdk_consumer is None:
        with _sdk_consumer_lock:
            if _sdk_consumer is None:
                _sdk_consumer = _start_audit_consumer(_sdk_audit_store)
    return _sdk_run_state_store, _sdk_audit_publisher


def _build_audit_publisher(fallback_store):
    """Wrap ``fallback_store`` with a Redis publisher if Redis pings, else
    return ``fallback_store`` so ``runner.audit_store.append`` keeps working.
    """
    try:
        from backend.audit.publisher import RedisAuditPublisher
        from backend.redis_bus import RedisBus
        bus = RedisBus()
        bus.ping()
    except Exception as e:
        print(f'[sdk-audit] Redis unavailable, writing audit rows synchronously to SQLite: {e}', flush=True)
        return fallback_store
    return RedisAuditPublisher(redis_client=bus.client, fallback_store=fallback_store)


def _start_audit_consumer(audit_store):
    """Start the daemon that drains ``audit:*`` Redis streams into SQLite.

    Returns the started :class:`AuditConsumer` (or ``None`` when Redis is
    unreachable, in which case the runner is already writing straight to
    SQLite via the publisher's fallback path).
    """
    try:
        from backend.audit.consumer import AuditConsumer
        from backend.redis_bus import RedisBus
        bus = RedisBus()
        bus.ping()
    except Exception as e:
        print(f'[sdk-audit] AuditConsumer not started (Redis unavailable): {e}', flush=True)
        return None
    consumer = AuditConsumer(redis_client=bus.client, store=audit_store)
    consumer.start()
    return consumer


def _start_sdk_sweeper(run_state_store, audit_store):
    interval = int(getattr(config, 'RUN_STATE_GC_INTERVAL_SECONDS', 300))
    completed_ttl = int(getattr(config, 'COMPLETED_RUN_TTL_SECONDS', 24 * 3600))
    audit_days = int(getattr(config, 'AUDIT_RETENTION_DAYS', 30))
    audit_purge_every = int(getattr(config, 'AUDIT_PURGE_INTERVAL_SECONDS', 3600))
    if interval <= 0:
        return

    def _loop():
        last_audit_purge = 0.0
        while True:
            try:
                run_state_store.gc_expired(completed_ttl_seconds=completed_ttl)
            except Exception as e:
                print(f'[sdk-sweeper] gc_expired error: {e}', flush=True)
            now = time.time()
            if audit_days > 0 and now - last_audit_purge >= audit_purge_every:
                try:
                    audit_store.purge_older_than(audit_days)
                    last_audit_purge = now
                except Exception as e:
                    print(f'[sdk-sweeper] audit purge error: {e}', flush=True)
            time.sleep(interval)

    thread = threading.Thread(target=_loop, name='sdk-sweeper', daemon=True)
    thread.start()


MAX_REQUEST_BODY_BYTES = int(getattr(config, 'MAX_REQUEST_BODY_BYTES', 8 * 1024 * 1024))

app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, 'SERVER_CORS_ORIGINS', ['*']),
    allow_methods=['*'],
    allow_headers=['*'],
)


async def _warmup_responses_runner():
    # Cold-start of the SDK runner drops the model's first response into
    # output[] = []. One in-process ping consumes that path so real user
    # requests start on the warm runner. ASGI transport avoids needing
    # to know the externally bound port.
    import asyncio
    await asyncio.sleep(0.5)
    try:
        import httpx
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://warmup', timeout=120) as client:
            async with client.stream(
                'POST', '/v1/responses',
                json={
                    'model': runtime_config.get_active_model(),
                    'input': 'ping',
                    'stream': True,
                    'store': False,
                },
            ) as resp:
                async for _ in resp.aiter_lines():
                    pass
        logger.info('[warmup] sdk runner warmed via /v1/responses ping')
    except Exception as exc:  # noqa: BLE001
        logger.warning('[warmup] sdk runner warmup failed: %s', exc)


@app.on_event('startup')
async def _on_startup_warmup():
    import asyncio
    asyncio.create_task(_warmup_responses_runner())


def _error_payload_from_detail(detail, default_code=''):
    if isinstance(detail, dict):
        payload = {
            'message': detail.get('message') or detail.get('detail') or '',
            'type': detail.get('type') or 'invalid_request_error',
            'code': detail.get('code') or default_code,
        }
        for key, value in detail.items():
            if key not in ('message', 'type', 'code', 'detail'):
                payload[key] = value
        return payload
    return {
        'message': '' if detail is None else str(detail),
        'type': 'invalid_request_error',
        'code': default_code,
    }


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException):
    return JSONResponse({'error': _error_payload_from_detail(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    try:
        message = '; '.join(
            f"{'.'.join(str(part) for part in err.get('loc') or [])}: {err.get('msg') or ''}".strip(': ')
            for err in exc.errors()
        ) or 'Invalid request'
    except Exception:
        message = 'Invalid request'
    return JSONResponse(
        {'error': {'message': message, 'type': 'invalid_request_error', 'code': 'validation_error'}},
        status_code=422,
    )


@app.middleware('http')
async def http_guards(request: Request, call_next):
    content_length = request.headers.get('content-length')
    if MAX_REQUEST_BODY_BYTES and content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    {'error': {'code': 'request_too_large', 'message': 'Request body is too large'}},
                    status_code=413,
                )
        except ValueError:
            pass
    response = await call_next(request)
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    return response


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
    if isinstance(exc, NoRegeneratableAnswerError):
        return HTTPException(
            status_code=409,
            detail={'code': 'no_regeneratable_answer', 'message': str(exc)},
        )
    if isinstance(exc, (WorkspaceViolation, ToolWorkspaceViolation)):
        return HTTPException(status_code=400, detail={'code': 'workspace_violation', 'message': str(exc)})
    return exc


def openai_error(message, code='invalid_request_error'):
    return {'error': {'message': message, 'type': 'invalid_request_error', 'code': code}}


def _pop_aliyun_credentials_response(body: dict):
    try:
        return pop_aliyun_credentials(body), None
    except AliyunCredentialError as exc:
        return None, JSONResponse(
            openai_error(str(exc), code='invalid_aliyun_credentials'),
            status_code=400,
        )


def _create_aliyun_profile_response(credentials: AliyunCredentials | None):
    if credentials is None:
        return None, None
    try:
        lease = write_temporary_profile(credentials)
    except AliyunConfigError as exc:
        logger.warning('Failed to prepare Aliyun runtime profile: %s', exc, exc_info=True)
        return None, JSONResponse(
            openai_error(str(exc), code='aliyun_config_error'),
            status_code=500,
        )
    logger.info('Prepared Aliyun runtime profile: profile=%s', lease.profile_name)
    return lease, None


def _cleanup_aliyun_profile(lease: AliyunProfileLease | None) -> None:
    if lease is None:
        return
    try:
        lease.cleanup()
    except Exception as exc:
        logger.warning(
            'Failed to cleanup Aliyun runtime profile: profile=%s error=%s',
            lease.profile_name,
            exc,
            exc_info=True,
        )


async def _with_aliyun_profile_cleanup(gen, lease: AliyunProfileLease | None):
    try:
        async for chunk in gen:
            yield chunk
    finally:
        aclose = getattr(gen, 'aclose', None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                logger.debug('Failed to close wrapped stream during Aliyun profile cleanup', exc_info=True)
        _cleanup_aliyun_profile(lease)


def stream_headers():
    return {
        'Cache-Control': 'no-cache, no-transform',
        'X-Accel-Buffering': 'no',
    }


def sse_encode(data=None, event=None, event_id=None, comment=None):
    if comment is not None:
        return f': {comment}\n\n'
    lines = []
    if event_id:
        lines.append(f'id: {event_id}')
    if event:
        lines.append(f'event: {event}')
    lines.append(f'data: {json.dumps(data, ensure_ascii=False, default=str)}')
    return '\n'.join(lines) + '\n\n'


def response_id():
    return f'resp_{uuid.uuid4().hex}'


_SSE_KEEPALIVE_INTERVAL_SECONDS = 15.0


async def _with_sse_keepalive(inner, interval: float = _SSE_KEEPALIVE_INTERVAL_SECONDS):
    """Yield from ``inner`` and inject SSE comment heartbeats when idle.

    Reverse proxies (nginx, ALB, Cloudflare) close idle SSE connections after
    30–60s. Long-running tool calls (``code_run`` with multi-minute scripts,
    LLM warmup) can stall the response stream past that threshold even though
    the run is healthy. We interleave ``: keepalive\\n\\n`` comment lines —
    SSE comments are silently discarded by every spec-compliant client and
    keep the underlying TCP connection active.

    Implementation uses a producer task draining ``inner`` into a queue; the
    consumer races ``queue.get()`` against the interval. Avoids the PEP 479
    pitfalls of awaiting a Task that raised ``StopAsyncIteration`` directly.
    """
    import asyncio as _asyncio
    queue: _asyncio.Queue = _asyncio.Queue()
    sentinel: object = object()

    async def _producer() -> None:
        try:
            async for chunk in inner:
                await queue.put(chunk)
        finally:
            await queue.put(sentinel)

    task = _asyncio.create_task(_producer())
    try:
        while True:
            try:
                item = await _asyncio.wait_for(queue.get(), timeout=interval)
            except _asyncio.TimeoutError:
                yield ': keepalive\n\n'
                continue
            if item is sentinel:
                return
            yield item
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except BaseException:  # noqa: BLE001 — swallow cancellation
            pass


def response_object(resp_id, model, status='completed', output=None, created_at=None, error=None, usage=None):
    payload = {
        'id': resp_id,
        'object': 'response',
        'created_at': created_at or int(time.time()),
        'status': status,
        'model': model,
        'output': output or [],
        'usage': dict(usage) if usage else None,
    }
    if error:
        payload['error'] = error
    return payload


def response_text_item(text):
    return {
        'id': f'msg_{uuid.uuid4().hex}',
        'type': 'message',
        'status': 'completed',
        'role': 'assistant',
        'content': [{'type': 'output_text', 'text': text or ''}],
    }


def response_function_call_item(event):
    raw_arguments = event.get('input')
    if isinstance(raw_arguments, str):
        arguments = raw_arguments
    else:
        arguments = json.dumps(raw_arguments or {}, ensure_ascii=False, default=str)
    return {
        'id': f'fc_{uuid.uuid4().hex}',
        'type': 'function_call',
        'status': 'completed',
        'call_id': event.get('tool_call_id') or '',
        'name': event.get('tool') or 'tool',
        'arguments': arguments,
    }


def response_function_output_item(event):
    output = event.get('content') or event.get('data')
    if not isinstance(output, str):
        output = json.dumps(output or {}, ensure_ascii=False, default=str)
    return {
        'id': f'fco_{uuid.uuid4().hex}',
        'type': 'function_call_output',
        'status': 'completed',
        'call_id': event.get('tool_call_id') or '',
        'output': output,
    }


def response_sse(event_type, data, sequence_number):
    payload = dict(data or {})
    payload.setdefault('type', event_type)
    payload.setdefault('sequence_number', sequence_number)
    return sse_encode(payload, event=event_type)


def content_text(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get('text'), str):
            return value.get('text') or ''
        if isinstance(value.get('content'), (str, list, dict)):
            return content_text(value.get('content'))
        return ''
    if isinstance(value, list):
        if any(isinstance(item, dict) and 'role' in item for item in value):
            text = last_user_text(value)
            if text:
                return text
            return content_text(value[-1].get('content') if value and isinstance(value[-1], dict) else '')
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get('type') in ('text', 'input_text', 'output_text') and isinstance(item.get('text'), str):
                parts.append(item.get('text') or '')
            else:
                parts.append(content_text(item))
        return '\n'.join(part for part in parts if part)
    return str(value)


def ensure_session(session_id=None, cwd=None):
    if session_id:
        sess = service.get_session(session_id, user_id=SERVER_USER_ID, cwd=cwd)
    else:
        sess = service.create_session(user_id=SERVER_USER_ID, cwd=cwd)
    if sess is None:
        raise HTTPException(status_code=404, detail='Session not found')
    return sess


def response_messages_from_input(raw_input):
    if raw_input is None:
        return []
    if isinstance(raw_input, str):
        return [{'role': 'user', 'content': raw_input}]
    if isinstance(raw_input, dict):
        role = raw_input.get('role') or 'user'
        return [{'role': role, 'content': content_text(raw_input.get('content', raw_input))}]
    if isinstance(raw_input, list):
        messages = []
        for item in raw_input:
            if isinstance(item, str):
                messages.append({'role': 'user', 'content': item})
            elif isinstance(item, dict) and 'role' in item:
                messages.append({'role': item.get('role') or 'user', 'content': content_text(item.get('content'))})
            else:
                text = content_text(item)
                if text:
                    messages.append({'role': 'user', 'content': text})
        return messages
    return [{'role': 'user', 'content': str(raw_input)}]


_RESPONSES_RESUME_ITEM_TYPES = {'function_call_output', 'mcp_approval_response'}


def has_responses_resume_input(raw_input):
    if not isinstance(raw_input, list):
        return False
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        item_type = item.get('type')
        if item_type not in _RESPONSES_RESUME_ITEM_TYPES:
            continue
        if item_type == 'function_call_output' and item.get('call_id'):
            return True
        if item_type == 'mcp_approval_response' and item.get('approval_request_id'):
            return True
    return False


def _history_message(role, content):
    text = content_text(content).strip()
    if not text:
        return None
    if role not in ('user', 'assistant', 'system', 'developer'):
        role = 'user'
    return {'role': role, 'content': text}


def _normalize_history_messages(messages):
    normalized = []
    for message in messages or []:
        if not isinstance(message, dict):
            item = _history_message('user', message)
        else:
            item = _history_message(message.get('role') or 'user', message.get('content', ''))
        if item:
            normalized.append(item)
    return normalized


def _session_history_messages(sess):
    if not sess:
        return []
    with sess._lock:
        ui_messages = list(sess.ui_msgs)
    return _normalize_history_messages(ui_messages)


def _assistant_message_from_response(response):
    text = _strip_thinking_blocks(_final_assistant_text((response or {}).get('output') or [])).strip()
    if not text:
        return None
    return {'role': 'assistant', 'content': text}


def _history_messages_from_response_record(record):
    history = _normalize_history_messages((record or {}).get('conversation_history') or [])
    assistant = _assistant_message_from_response((record or {}).get('response') or {})
    if assistant:
        last = history[-1] if history else None
        if not last or last.get('role') != 'assistant' or last.get('content') != assistant.get('content'):
            history.append(assistant)
    return history


def _compose_runner_input(history_messages, current_messages):
    history = _normalize_history_messages(history_messages)
    current = _normalize_history_messages(current_messages)
    if not history and len(current) == 1 and current[0].get('role') == 'user':
        return current[0].get('content') or ''
    return history + current


def _last_user_content(messages):
    return next(
        ((message.get('content') or '').strip() for message in reversed(messages or []) if message.get('role') == 'user'),
        '',
    )


_PAIRAG_NAMESPACE_KEY = 'pairag'
_FINAL_REPORT_FLAG = 'is_final_report'
_PROCESS_REASONING_FLAG = 'is_process_reasoning'


def _pairag_flag(item, flag):
    if not isinstance(item, dict):
        return False
    metadata = item.get('metadata')
    if not isinstance(metadata, dict):
        return False
    namespace = metadata.get(_PAIRAG_NAMESPACE_KEY)
    if not isinstance(namespace, dict):
        return False
    return bool(namespace.get(flag))


def _is_final_report_output_item(item):
    return _pairag_flag(item, _FINAL_REPORT_FLAG)


def _is_process_reasoning_output_item(item):
    return (
        isinstance(item, dict) and item.get('type') == 'reasoning'
    ) or _pairag_flag(item, _PROCESS_REASONING_FLAG)


def _reasoning_output_text(item):
    parts = []
    for block in item.get('content') or []:
        if not isinstance(block, dict):
            continue
        if block.get('type') in ('reasoning_text', 'summary_text', 'text', 'output_text'):
            text = block.get('text') or ''
            if text:
                parts.append(text)
    return ''.join(parts)


def _final_assistant_text(output_list):
    """Extract assistant text from a Responses-API output[] list.

    When the SDK final answer contract is present, the structured final report
    is authoritative and earlier protocol-only messages are ignored.
    """
    parts = []
    after_tool_parts = []
    report_parts = []
    last_tool_output = -1
    for index, item in enumerate(output_list or []):
        if isinstance(item, dict) and item.get('type') == 'function_call_output':
            last_tool_output = index
    for item_index, item in enumerate(output_list or []):
        if not isinstance(item, dict) or item.get('type') != 'message':
            continue
        if _is_process_reasoning_output_item(item):
            continue
        item_parts = []
        for block in item.get('content') or []:
            if not isinstance(block, dict):
                continue
            if block.get('type') in ('output_text', 'text'):
                text = block.get('text') or ''
                if text:
                    item_parts.append(text)
        item_text = ''.join(item_parts)
        if not item_text:
            continue
        if _is_final_report_output_item(item):
            report_parts.append(item_text)
        else:
            parts.append(item_text)
            if last_tool_output >= 0 and item_index > last_tool_output:
                after_tool_parts.append(item_text)
    if report_parts:
        return ''.join(report_parts)
    if last_tool_output >= 0 and after_tool_parts:
        return ''.join(after_tool_parts)
    return ''.join(parts)


_THINK_TAGS = (
    ('<forcing_skill_activation>', '</forcing_skill_activation>'),
    ('<clinical-thinking>', '</clinical-thinking>'),
    ('<clinical_thinking>', '</clinical_thinking>'),
    ('<taking-action>', '</taking-action>'),
    ('<taking_action>', '</taking_action>'),
    ('<skill-context>', '</skill-context>'),
    ('<skill_context>', '</skill_context>'),
    ('<thinking>', '</thinking>'),
    ('<checking>', '</checking>'),
    ('<taking>', '</taking>'),
    ('<working>', '</working>'),
)
_DISCARD_THINK_OPEN_TAGS = {'<forcing_skill_activation>'}
_INTERNAL_TOOL_NAMES = {'update_working_checkpoint', 'update_todo', 'start_long_term_update', 'final_report'}


def _tool_kind(name):
    if name == 'code_run':
        return 'execute'
    if name in ('file_read',):
        return 'read'
    if name in ('file_write', 'file_patch'):
        return 'write'
    return 'tool'


def _strip_thinking_blocks(text):
    """Remove private reasoning blocks from text. Used so the
    persisted assistant ``content`` is the user-facing answer; thinking is
    surfaced separately via ``events`` (thought_start/delta/done)."""
    if not text:
        return ''
    out = []
    pos = 0
    while True:
        found = _find_next_think_tag(text, pos)
        if found is None:
            out.append(text[pos:])
            break
        i, open_tag, close_tag = found
        out.append(text[pos:i])
        j = text.find(close_tag, i + len(open_tag))
        if j == -1:
            break
        pos = j + len(close_tag)
    return ''.join(out)


def _find_next_think_tag(text, pos=0):
    best = None
    for open_tag, close_tag in _THINK_TAGS:
        index = text.find(open_tag, pos)
        if index != -1 and (best is None or index < best[0]):
            best = (index, open_tag, close_tag)
    return best


def _thought_updates_from_text(text, thought_id):
    updates = [{
        'sessionUpdate': 'thought_start', 'thoughtId': thought_id,
        'title': 'Thinking', 'status': 'in_progress',
    }]
    pos = 0
    while pos < len(text):
        found = _find_next_think_tag(text, pos)
        if found is None:
            chunk = text[pos:]
            if chunk:
                updates.append({
                    'sessionUpdate': 'thought_delta', 'thoughtId': thought_id,
                    'content': {'type': 'text', 'text': chunk},
                })
            break
        i, open_tag, close_tag = found
        if i > pos:
            updates.append({
                'sessionUpdate': 'thought_delta', 'thoughtId': thought_id,
                'content': {'type': 'text', 'text': text[pos:i]},
            })
        j = text.find(close_tag, i + len(open_tag))
        if j == -1:
            inner = text[i + len(open_tag):]
            end = len(text)
        else:
            inner = text[i + len(open_tag):j]
            end = j + len(close_tag)
        if open_tag not in _DISCARD_THINK_OPEN_TAGS and inner:
            updates.append({
                'sessionUpdate': 'thought_delta', 'thoughtId': thought_id,
                'content': {'type': 'text', 'text': inner},
            })
        pos = end
    updates.append({
        'sessionUpdate': 'thought_done', 'thoughtId': thought_id,
        'status': 'completed',
    })
    return updates


def _agent_updates_from_output(output_list):
    """Build frontend-shaped ``AgentUpdate`` dicts from Responses API output.

    Mirrors what the frontend's ``responsesEventToUpdates`` produces during
    live streaming, but reconstructed from the persisted ``output`` list.
    Used so refreshing a session restores the per-step card UI (thinking,
    tool calls, results) instead of only the final answer text.

    Inline private reasoning blocks within message text are split
    out into ``thought_*`` events; remaining prose becomes
    ``agent_message_chunk`` updates.
    """
    updates = []
    think_counter = 0
    for item in output_list or []:
        if not isinstance(item, dict):
            continue
        t = item.get('type')
        if t == 'reasoning':
            text = _reasoning_output_text(item)
            if text:
                think_counter += 1
                updates.extend(_thought_updates_from_text(text, f'persisted-process-think-{think_counter}'))
            continue
        if t == 'message':
            for block in item.get('content') or []:
                if not isinstance(block, dict):
                    continue
                if block.get('type') not in ('output_text', 'text'):
                    continue
                text = block.get('text') or ''
                if not text:
                    continue
                if _is_process_reasoning_output_item(item):
                    think_counter += 1
                    updates.extend(_thought_updates_from_text(text, f'persisted-process-think-{think_counter}'))
                    continue
                pos = 0
                while pos < len(text):
                    found = _find_next_think_tag(text, pos)
                    if found is None:
                        chunk = text[pos:]
                        if chunk:
                            updates.append({
                                'sessionUpdate': 'agent_message_chunk',
                                'content': {'type': 'text', 'text': chunk},
                            })
                        break
                    i, open_tag, close_tag = found
                    if i > pos:
                        updates.append({
                            'sessionUpdate': 'agent_message_chunk',
                            'content': {'type': 'text', 'text': text[pos:i]},
                        })
                    j = text.find(close_tag, i + len(open_tag))
                    if j == -1:
                        inner = text[i + len(open_tag):]
                        end = len(text)
                    else:
                        inner = text[i + len(open_tag):j]
                        end = j + len(close_tag)
                    if open_tag in _DISCARD_THINK_OPEN_TAGS:
                        pos = end
                        continue
                    think_counter += 1
                    tid = f'persisted-think-{think_counter}'
                    updates.append({
                        'sessionUpdate': 'thought_start', 'thoughtId': tid,
                        'title': 'Thinking', 'status': 'in_progress',
                    })
                    if inner:
                        updates.append({
                            'sessionUpdate': 'thought_delta', 'thoughtId': tid,
                            'content': {'type': 'text', 'text': inner},
                        })
                    updates.append({
                        'sessionUpdate': 'thought_done', 'thoughtId': tid,
                        'status': 'completed',
                    })
                    pos = end
        elif t == 'function_call':
            call_id = item.get('call_id') or item.get('id') or ''
            name = item.get('name') or ''
            args = item.get('arguments') or ''
            kind = _tool_kind(name)
            hidden = name in _INTERNAL_TOOL_NAMES
            parsed_args = None
            if isinstance(args, str) and args.strip():
                try:
                    parsed_args = json.loads(args)
                except (TypeError, ValueError):
                    parsed_args = None
            tool_call_evt = {
                'sessionUpdate': 'tool_call', 'toolCallId': call_id,
                'title': name, 'name': name, 'kind': kind,
                'status': 'completed', 'hidden': hidden,
            }
            if isinstance(parsed_args, dict):
                tool_call_evt['input'] = parsed_args
            updates.append(tool_call_evt)
            if isinstance(args, str) and args and not isinstance(parsed_args, dict):
                # Args were unparseable JSON; surface as inputDraft via the
                # delta event so the UI can still show the raw string.
                updates.append({
                    'sessionUpdate': 'tool_call_delta', 'toolCallId': call_id,
                    'index': 0, 'argumentsText': args,
                    'status': 'completed',
                })
        elif t == 'function_call_output':
            call_id = item.get('call_id') or ''
            output = item.get('output') or ''
            update = {
                'sessionUpdate': 'tool_call_update', 'toolCallId': call_id,
                'status': 'completed',
            }
            if isinstance(output, str):
                update['content'] = {'type': 'text', 'text': output}
                stripped = output.strip()
                if stripped.startswith(('{', '[')):
                    try:
                        update['data'] = json.loads(stripped)
                    except (TypeError, ValueError):
                        pass
            else:
                update['data'] = output
            updates.append(update)
    return updates


def _review_history_from_session(sess):
    history = []
    for msg in getattr(sess, 'ui_msgs', []) or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role') or ''
        if role not in {'user', 'assistant', 'system', 'tool'}:
            continue
        text = content_text(msg.get('content')).strip()
        if not text:
            continue
        history.append({'role': role, 'content': text})
    return history


def _schedule_background_review_after_persist(
    sess,
    *,
    final_status='',
    archive_path='',
    run_id='',
    task_text='',
    active_skill='',
    allow_background_review=True,
):
    if not allow_background_review or final_status != 'completed' or not archive_path:
        return
    memory_scope = getattr(sess, 'memory_scope', None)
    memory_root = getattr(memory_scope, 'root', '') or ''
    if not memory_root:
        logger.info(
            'Background memory review skipped: reason=missing_memory_root session=%s run=%s',
            getattr(sess, 'sid', ''),
            run_id,
        )
        return
    user_id = getattr(sess, 'user_id', SERVER_USER_ID) or SERVER_USER_ID
    try:
        schedule_background_memory_review(
            user_id=user_id,
            session_id=getattr(sess, 'sid', '') or '',
            run_id=run_id or '',
            task_text=task_text or '',
            llm_history=_review_history_from_session(sess),
            memory_root=memory_root,
            active_skill=active_skill or '',
            final_status=final_status,
            long_term_enabled=long_term_memory_enabled(user_id),
            archive_path=archive_path,
        )
    except Exception as exc:
        logger.warning(
            'Background memory review scheduling failed: session=%s run=%s error=%s',
            getattr(sess, 'sid', ''),
            run_id,
            exc,
            exc_info=True,
        )


def _append_turn_to_session(
    sess,
    *,
    user_text,
    assistant_text,
    assistant_events=None,
    final_status='',
    run_id='',
    allow_background_review=True,
    active_skill='',
    task_text='',
):
    """Append user + (optional) assistant message to sess.ui_msgs and persist.

    Without this, /v1/responses runs never write to the legacy ``ui_messages``
    column, so ``GET /v1/sessions/{id}`` returns ``messages: []`` and the
    frontend wipes streamed answers on refresh.

    ``assistant_events`` carries the full process trace (thinking / tool calls
    / outputs) so a session reload re-renders the Agent step cards instead
    of dropping everything but the final text.
    """
    if not sess:
        return ''
    new_msgs = []
    if user_text:
        new_msgs.append({'role': 'user', 'content': user_text})
    if assistant_text or assistant_events:
        msg = {'role': 'assistant', 'content': assistant_text or ''}
        if assistant_events:
            msg['events'] = assistant_events
        new_msgs.append(msg)
    if not new_msgs:
        return ''
    with sess._lock:
        sess.ui_msgs = list(sess.ui_msgs) + new_msgs
    archive_path = ''
    try:
        sess.save()
        archive_path = _archive_session_for_replay(sess)
    except Exception as exc:
        print(f'[ui_msgs] save failed for session {sess.sid}: {exc}', flush=True)
        return ''
    _schedule_background_review_after_persist(
        sess,
        final_status=final_status,
        archive_path=archive_path,
        run_id=run_id,
        task_text=task_text or user_text,
        active_skill=active_skill,
        allow_background_review=allow_background_review,
    )
    return archive_path


def _archive_session_for_replay(sess):
    try:
        loaded = service.store.load(sess.sid, user_id=sess.user_id)
        if loaded:
            return archive_session_record(
                loaded,
                sess.memory_scope.archive_dir,
                created_after=SESSION_ARCHIVE_CREATED_AFTER,
            )
    except Exception as exc:
        print(f'[session_archive] save failed for session {sess.sid}: {exc}', flush=True)
    return ''


def save_response_record(resp_id, response, conversation_history, instructions='', session_id='', conversation='', store=True):
    if not store:
        return
    service.store.save_response(
        resp_id,
        response,
        conversation_history=conversation_history,
        instructions=instructions,
        session_id=session_id,
        conversation=conversation,
        user_id=SERVER_USER_ID,
    )


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/health/detailed')
def health_detailed():
    checks = {'api': 'ok', 'sqlite': 'unknown', 'redis': 'disabled', 'runner': RUNNER_BACKEND}
    try:
        with service.store._connect() as conn:
            conn.execute('SELECT 1').fetchone()
        checks['sqlite'] = 'ok'
    except Exception as e:
        checks['sqlite'] = f'error: {e}'
    status = 'ok' if all(value in ('ok', 'disabled', RUNNER_BACKEND) for value in checks.values()) else 'degraded'
    return {
        'status': status,
        'runner_backend': RUNNER_BACKEND,
        'model': runtime_config.get_active_model(),
        'checks': checks,
    }


@app.get('/v1/models')
def models():
    active = runtime_config.get_active_model()
    configured = getattr(config, 'MODEL', 'qwen-plus')
    aliases = list(getattr(config, 'MODEL_ALIASES', ['mini-agent', 'agent']) or [])
    ids = []
    for model_id in [active, configured, *aliases]:
        if model_id and model_id not in ids:
            ids.append(model_id)
    return {
        'object': 'list',
        'active_model': active,
        'data': [
            {
                'id': model_id,
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'pai-rag',
                'active': model_id == active,
            }
            for model_id in ids
        ],
    }


@app.post('/v1/models/active')
async def set_active_model_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    try:
        new_model = runtime_config.set_active_model(body.get('model'))
    except ValueError as exc:
        return JSONResponse(openai_error(str(exc)), status_code=400)
    return {'active_model': new_model}


@app.get('/v1/skills')
def skills():
    scope = handler_memory_scope(SERVER_USER_ID)
    return skills_inventory(ROOT, scope.root)


@app.get('/v1/admin/pool')
def admin_pool():
    """Per-worker view of the LLM credential pool. Read-only, redacts keys
    to a 4-char tail. State is in-process so each worker reports its own
    eviction/cooldown state; agreement across workers is not guaranteed."""
    return provider_pool.snapshot()


async def _sdk_response_stream(*, body, model, model_override, instructions, conversation,
                               session_id, cwd, store, previous_response_id, allow_hitl=False,
                               tool_env=None):
    """Drive the SDK runner and emit Responses-API SSE chunks."""
    from backend.agents_sdk import runner as sdk_runner
    from backend.agents_sdk.hitl import ResumePayload
    from backend.agents_sdk.runtime_setup import ensure_sdk_runtime
    from backend.tools.wrappers import build_tool_list
    from tools import GenericHandler

    ensure_sdk_runtime()
    run_state_store, audit_store = _sdk_stores()

    sequence = 0

    def emit(event_type, data):
        nonlocal sequence
        sequence += 1
        return response_sse(event_type, data, sequence)

    raw_input = body.get('input')
    resume_payload = None
    input_text = None
    input_items = None
    user_text_for_persist = ''
    conversation_history_for_record = []
    previous_response_record = None
    resume_input = has_responses_resume_input(raw_input)

    if previous_response_id and resume_input:
        try:
            items = raw_input if isinstance(raw_input, list) else []
            resume_payload = ResumePayload.from_responses_input(items)
            user_text_for_persist = resume_payload.answer.strip()
        except ValueError as exc:
            yield emit('response.failed', {
                'response_id': '', 'error': {'message': str(exc), 'code': 'invalid_resume'},
            })
            yield 'data: [DONE]\n\n'
            return
    elif previous_response_id:
        previous_response_record = service.store.load_response(previous_response_id, user_id=SERVER_USER_ID)
        if previous_response_record is None:
            yield emit('response.failed', {
                'response_id': previous_response_id,
                'error': {'message': f'Response not found: {previous_response_id}', 'code': 'response_not_found'},
            })
            yield 'data: [DONE]\n\n'
            return
        if not session_id:
            session_id = previous_response_record.get('session_id') or ''
        if not instructions:
            instructions = previous_response_record.get('instructions') or ''
        if not conversation:
            conversation = previous_response_record.get('conversation') or ''

    if resume_payload is None:
        input_messages = response_messages_from_input(raw_input)
        current_messages = _normalize_history_messages(input_messages)
        user_text_for_persist = _last_user_content(current_messages)
        if not user_text_for_persist:
            yield emit('response.failed', {
                'response_id': '', 'error': {'message': 'No user message in input', 'code': 'invalid_input'},
            })
            yield 'data: [DONE]\n\n'
            return

    try:
        sess = ensure_session(session_id or None, cwd=cwd)
    except WorkspaceViolation as e:
        raise backend_error(e) from e

    if resume_payload is None:
        if previous_response_record is not None:
            history_messages = _history_messages_from_response_record(previous_response_record)
            if not history_messages:
                history_messages = _session_history_messages(sess)
        else:
            history_messages = _session_history_messages(sess)
        conversation_history_for_record = _normalize_history_messages(history_messages + current_messages)
        runner_input = _compose_runner_input(history_messages, current_messages)
        if isinstance(runner_input, list):
            input_items = runner_input
        else:
            input_text = runner_input

    handler = GenericHandler(
        cwd=sess.workspace_path or sess.cwd,
        mini_agent_root=ROOT,
        workspace_root=sess.workspace_root,
        readonly_roots=sess.readonly_roots,
        run_env=tool_env,
    )
    extras = {'handler': handler, 'session': sess}

    created_at = int(time.time())
    response_id_holder = {'id': previous_response_id if resume_payload is not None else ''}
    yielded_created = False
    visible_text_emitted = False

    try:
        async for frame in sdk_runner.stream_responses_run(
            state_store=run_state_store,
            audit_store=audit_store,
            session_id=sess.sid,
            user_id=SERVER_USER_ID,
            model=model,
            cwd=sess.workspace_path or sess.cwd,
            tools=build_tool_list(scope='main'),
            input_text=input_text,
            input_items=input_items,
            previous_response_id=previous_response_id if resume_payload is not None else None,
            resume=resume_payload,
            instructions_override=None,
            max_turns=int(getattr(config, 'MAX_TURNS', 40)),
            extras=extras,
            allow_hitl=allow_hitl,
        ):
            if frame.chunk is not None:
                rid = frame.chunk.get('response_id')
                if rid:
                    response_id_holder['id'] = rid
                if not yielded_created and response_id_holder['id']:
                    yielded_created = True
                    yield emit('response.created', response_object(
                        response_id_holder['id'], model, status='in_progress', created_at=created_at,
                    ))
                if frame.chunk.get('type') == 'response.output_text.delta' and frame.chunk.get('delta'):
                    visible_text_emitted = True
                yield emit(frame.chunk['type'], frame.chunk)
            if frame.terminal:
                final = frame.response_object or {}
                rid = final.get('id') or response_id_holder['id']
                if not yielded_created:
                    yielded_created = True
                    yield emit('response.created', response_object(
                        rid, model, status='in_progress', created_at=created_at,
                    ))
                final.setdefault('created_at', created_at)
                final.setdefault('model', model)
                if (final.get('status') == 'completed' and not visible_text_emitted):
                    final_text = _final_assistant_text(final.get('output') or [])
                    if final_text:
                        yield emit('response.output_text.delta', {
                            'type': 'response.output_text.delta',
                            'response_id': rid,
                            'delta': final_text,
                            'output_index': 0,
                            'content_index': 0,
                        })
                        visible_text_emitted = True
                if final.get('status') == 'requires_action':
                    yield emit('response.requires_action', final)
                    # Emit a standard Responses API terminator so strict clients
                    # (OpenAI SDK readers) that only look for completed/failed/
                    # incomplete don't hang. PAI-RAG-aware clients act on the
                    # ``response.requires_action`` envelope above; this is the
                    # graceful close for everyone else.
                    incomplete_payload = dict(final)
                    incomplete_payload['status'] = 'incomplete'
                    incomplete_payload['incomplete_details'] = {'reason': 'requires_action'}
                    yield emit('response.incomplete', incomplete_payload)
                elif final.get('status') == 'failed':
                    yield emit('response.failed', final)
                else:
                    yield emit('response.completed', final)

                # Persist user + assistant turns into ui_msgs so refresh /
                # session detail reads back the streamed conversation. Without
                # this the SQLite ui_messages column stays empty and the
                # frontend wipes streamed answers when it refetches the
                # session after [DONE].
                user_msg = user_text_for_persist
                final_status = final.get('status')
                output_list = final.get('output') or []
                assistant_events = None
                if final_status == 'completed':
                    raw_text = _final_assistant_text(output_list)
                    assistant_msg = _strip_thinking_blocks(raw_text).strip()
                    assistant_events = _agent_updates_from_output(output_list)
                elif final_status == 'failed':
                    err_msg = ''
                    err_obj = final.get('error') or {}
                    if isinstance(err_obj, dict):
                        err_msg = (err_obj.get('message') or '').strip()
                    assistant_msg = f'**Error:** {err_msg}' if err_msg else ''
                else:
                    assistant_msg = ''
                    if final_status == 'requires_action':
                        # Persist the partial trace (thinking + tool calls so far)
                        # so an HITL pause survives refresh — the ask_user card is
                        # already reconstructed from pending_hitl, but the
                        # preceding reasoning would otherwise be lost.
                        assistant_events = _agent_updates_from_output(output_list) or None
                _append_turn_to_session(
                    sess, user_text=user_msg, assistant_text=assistant_msg,
                    assistant_events=assistant_events,
                    final_status=final_status,
                    run_id=rid,
                    allow_background_review=store,
                    active_skill=(getattr(handler, 'working', {}) or {}).get('active_skill', ''),
                    task_text=user_msg,
                )

                if store and rid:
                    if resume_payload is not None and previous_response_id:
                        previous_record = service.store.load_response(previous_response_id, user_id=SERVER_USER_ID) or {}
                        history = _normalize_history_messages(previous_record.get('conversation_history') or [])
                    else:
                        history = conversation_history_for_record
                    save_response_record(
                        rid, final, history, instructions=instructions,
                        session_id=sess.sid, conversation=conversation, store=store,
                    )
                yield 'data: [DONE]\n\n'
                return
    except Exception as exc:
        yield emit('response.failed', response_object(
            response_id_holder['id'], model, status='failed',
            created_at=created_at, error={'message': str(exc)},
        ))
        yield 'data: [DONE]\n\n'


@app.post('/v1/responses')
async def create_response(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    legacy_fields = [key for key in ('session_id', 'conversation_history', 'messages') if key in body]
    if legacy_fields:
        joined = ', '.join(legacy_fields)
        return JSONResponse(openai_error(
            f'Unsupported legacy field(s): {joined}; use input with conversation or previous_response_id',
            code='unsupported_legacy_field',
        ), status_code=400)
    if request.headers.get('x-session-id'):
        return JSONResponse(openai_error(
            'X-Session-Id is no longer supported; use the conversation request field',
            code='unsupported_legacy_header',
        ), status_code=400)
    aliyun_credentials, error_response = _pop_aliyun_credentials_response(body)
    if error_response is not None:
        return error_response

    raw_model = body.get('model')
    model_override = None
    if raw_model:
        if not isinstance(raw_model, str):
            return JSONResponse(openai_error("'model' must be a string"), status_code=400)
        try:
            model_override = runtime_config.validate_model_name(raw_model)
        except ValueError as exc:
            return JSONResponse(openai_error(str(exc)), status_code=400)
    model = model_override or runtime_config.get_active_model()
    stream = bool(body.get('stream'))
    cwd = body.get('cwd')
    instructions = body.get('instructions') or ''
    conversation = body.get('conversation') or ''
    store = bool(body.get('store', True))
    previous_response_id = body.get('previous_response_id') or ''
    session_id_for_sdk = ''
    if not previous_response_id and conversation:
        previous_response_id = service.store.latest_response_for_conversation(
            conversation, user_id=SERVER_USER_ID,
        ) or ''
        if not previous_response_id:
            try:
                if service.store.session_exists(conversation):
                    session_id_for_sdk = conversation
            except ValueError:
                session_id_for_sdk = ''

    run_state_store, _ = _sdk_stores()
    resume_input_for_hitl_default = False
    if previous_response_id:
        row = run_state_store.get(previous_response_id, user_id=SERVER_USER_ID)
        resume_input = has_responses_resume_input(body.get('input'))
        resume_input_for_hitl_default = resume_input
        if resume_input:
            if row is None:
                return JSONResponse(openai_error(
                    f'Response not found or not resumable: {previous_response_id}',
                    code='response_not_found'), status_code=404)
            if row.get('status') != 'requires_action':
                return JSONResponse(openai_error(
                    f'Response is not awaiting tool output: {previous_response_id}',
                    code='not_resumable'), status_code=409)
            session_id_for_sdk = row['session_id']
        elif row is not None and row.get('status') == 'requires_action':
            return JSONResponse(openai_error(
                'Response requires function_call_output input to resume',
                code='invalid_resume'), status_code=400)
        else:
            record = service.store.load_response(previous_response_id, user_id=SERVER_USER_ID)
            if record is None:
                return JSONResponse(openai_error(
                    f'Response not found: {previous_response_id}',
                    code='response_not_found'), status_code=404)
            session_id_for_sdk = record.get('session_id') or ''
    aliyun_profile, error_response = _create_aliyun_profile_response(aliyun_credentials)
    if error_response is not None:
        return error_response
    gen = _sdk_response_stream(
        body=body, model=model, model_override=model_override,
        instructions=instructions, conversation=conversation,
        session_id=session_id_for_sdk, cwd=cwd, store=store,
        previous_response_id=previous_response_id,
        allow_hitl=bool(body.get('allow_hitl', resume_input_for_hitl_default)),
        tool_env=aliyun_profile.env if aliyun_profile is not None else None,
    )
    if stream:
        gen = _with_aliyun_profile_cleanup(gen, aliyun_profile)
        return StreamingResponse(_with_sse_keepalive(gen), media_type='text/event-stream',
                                 headers=stream_headers())
    try:
        return await _collect_responses_completion(gen)
    finally:
        _cleanup_aliyun_profile(aliyun_profile)


@app.get('/v1/responses/{resp_id}')
def get_response(resp_id: str):
    record = service.store.load_response(resp_id, user_id=SERVER_USER_ID)
    if record is None:
        return JSONResponse(openai_error(f'Response not found: {resp_id}', code='response_not_found'), status_code=404)
    return record.get('response') or {}


@app.delete('/v1/responses/{resp_id}')
def delete_response(resp_id: str):
    deleted = service.store.delete_response(resp_id, user_id=SERVER_USER_ID)
    if not deleted:
        return JSONResponse(openai_error(f'Response not found: {resp_id}', code='response_not_found'), status_code=404)
    return {'id': resp_id, 'object': 'response.deleted', 'deleted': True}


@app.post('/v1/responses/{resp_id}/cancel')
def cancel_response(resp_id: str):
    """OpenAI Responses-API standard cancel by ``response_id``.

    Resolution order for ``session_id`` (because a response_id can refer to a
    live, mid-flight run that hasn't been persisted yet):

    1. **In-flight registry** (``runner.session_for_inflight_response``) —
       populated when a run starts, cleared at completion. Catches active runs.
    2. **Persisted response record** (``service.store.load_response``) —
       populated at terminal time. Catches completed/paused runs that strict
       clients still want to mark cancelled.

    Returns the (possibly stale) response object with ``status='cancelled'``
    so the wire shape matches OpenAI's cancel response.
    """
    from backend.agents_sdk import runner as sdk_runner

    sid = sdk_runner.session_for_inflight_response(resp_id) or ''
    record = None
    if not sid:
        record = service.store.load_response(resp_id, user_id=SERVER_USER_ID)
        if record is None:
            return JSONResponse(
                openai_error(f'Response not found: {resp_id}', code='response_not_found'),
                status_code=404,
            )
        sid = record.get('session_id') or ''
    if sid:
        service.cancel_session(sid, user_id=SERVER_USER_ID)
    response_obj = dict((record or {}).get('response') or {})
    response_obj['id'] = resp_id
    response_obj['object'] = 'response'
    response_obj['status'] = 'cancelled'
    return response_obj


async def _sdk_chat_stream(*, body, model, cwd, allow_hitl=False, tool_env=None):
    """Drive the SDK runner and emit public Chat Completions SSE chunks."""
    from backend.agents_sdk import event_bridge as _bridge
    from backend.agents_sdk import runner as sdk_runner
    from backend.agents_sdk.hitl import ResumePayload
    from backend.agents_sdk.runtime_setup import ensure_sdk_runtime
    from backend.tools.wrappers import build_tool_list
    from tools import GenericHandler

    ensure_sdk_runtime()
    run_state_store, audit_store = _sdk_stores()

    completion_id = f'chatcmpl-{uuid.uuid4().hex}'
    messages = body.get('messages') or []

    resume_payload = None
    resume_row = None
    if messages and isinstance(messages[-1], dict) and messages[-1].get('role') == 'tool':
        try:
            resume_payload = ResumePayload.from_chat_message(messages[-1])
        except ValueError as exc:
            yield sse_encode({'error': {'message': str(exc), 'code': 'invalid_resume'}})
            yield 'data: [DONE]\n\n'
            return
        resume_row = run_state_store.find_by_pending_call_id(resume_payload.call_id)
        if resume_row is None:
            yield sse_encode({'error': {
                'message': f'No paused run found for tool_call_id={resume_payload.call_id!r}',
                'code': 'resume_not_found',
            }})
            yield 'data: [DONE]\n\n'
            return

    if resume_payload is None:
        user_text = last_user_text(messages)
        if not user_text.strip():
            yield sse_encode({'error': {'message': 'messages must include a non-empty user message',
                                        'code': 'invalid_input'}})
            yield 'data: [DONE]\n\n'
            return
        try:
            sess = ensure_session(None, cwd=cwd)
        except WorkspaceViolation as e:
            raise backend_error(e) from e
        session_id = sess.sid
        cwd_resolved = sess.workspace_path or sess.cwd
        readonly = sess.readonly_roots
        workspace_root = sess.workspace_root
        chat_messages = _normalize_history_messages(response_messages_from_input(messages))
        runner_input = _compose_runner_input([], chat_messages)
        input_items = runner_input if isinstance(runner_input, list) else None
        input_text = None if isinstance(runner_input, list) else runner_input
        sess_for_persist = sess
    else:
        session_id = resume_row['session_id']
        loaded = service.store.load(session_id, user_id=SERVER_USER_ID) or {}
        cwd_resolved = loaded.get('workspace_path') or cwd or os.getcwd()
        readonly = []
        workspace_root = loaded.get('workspace_path') or None
        input_items = None
        input_text = None
        try:
            sess_for_persist = ensure_session(session_id, cwd=cwd_resolved)
        except Exception as exc:
            logger.warning(
                'Failed to restore session for chat resume persistence: session=%s error=%s',
                session_id,
                exc,
                exc_info=True,
            )
            sess_for_persist = None

    handler = GenericHandler(
        cwd=cwd_resolved, mini_agent_root=ROOT,
        workspace_root=workspace_root, readonly_roots=readonly,
        run_env=tool_env,
    )
    extras = {'handler': handler}
    if sess_for_persist is not None:
        extras['session'] = sess_for_persist

    yield sse_encode(_bridge.chat_role_chunk(model=model, completion_id=completion_id))

    captured_usage: dict = {}
    try:
        async for frame in sdk_runner.stream_responses_run(
            state_store=run_state_store,
            audit_store=audit_store,
            session_id=session_id,
            user_id=SERVER_USER_ID,
            model=model,
            cwd=cwd_resolved,
            tools=build_tool_list(scope='main'),
            input_text=input_text,
            input_items=input_items,
            previous_response_id=resume_row['id'] if resume_row else None,
            resume=resume_payload,
            instructions_override=None,
            max_turns=int(getattr(config, 'MAX_TURNS', 40)),
            extras=extras,
            allow_hitl=allow_hitl,
        ):
            if frame.chunk is not None and frame.chunk.get('type') == 'response.output_text.delta':
                delta = frame.chunk.get('delta') or ''
                if delta:
                    yield sse_encode({
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': int(time.time()),
                        'model': model,
                        'choices': [{'index': 0, 'delta': {'content': delta}, 'finish_reason': None}],
                    })
            if frame.terminal:
                final = frame.response_object or {}
                if final.get('status') == 'requires_action' and frame.interruption is not None:
                    yield sse_encode(_bridge.chat_pause_chunk(
                        frame.interruption, model=model, completion_id=completion_id,
                    ))
                elif final.get('status') == 'failed':
                    err = final.get('error') or {}
                    yield sse_encode({'error': {
                        'message': err.get('message') or 'Run failed',
                        'code': 'run_failed',
                    }})
                else:
                    captured_usage.update(final.get('usage') or {})
                    yield sse_encode(_bridge.chat_done_chunk(
                        model=model, completion_id=completion_id,
                        usage=captured_usage or None,
                    ))

                if sess_for_persist is not None:
                    user_msg = (resume_payload.answer if resume_payload else (user_text or '')).strip()
                    final_status = final.get('status')
                    output_list = final.get('output') or []
                    assistant_events = None
                    if final_status == 'completed':
                        raw_text = _final_assistant_text(output_list)
                        assistant_msg = _strip_thinking_blocks(raw_text).strip()
                        assistant_events = _agent_updates_from_output(output_list)
                    elif final_status == 'failed':
                        err_obj = final.get('error') or {}
                        err_text = (err_obj.get('message') or '').strip() if isinstance(err_obj, dict) else ''
                        assistant_msg = f'**Error:** {err_text}' if err_text else ''
                    else:
                        assistant_msg = ''
                        if final_status == 'requires_action':
                            assistant_events = _agent_updates_from_output(output_list) or None
                    _append_turn_to_session(
                        sess_for_persist,
                        user_text=user_msg,
                        assistant_text=assistant_msg,
                        assistant_events=assistant_events,
                        final_status=final_status,
                        run_id=final.get('id') or completion_id,
                        active_skill=(getattr(handler, 'working', {}) or {}).get('active_skill', ''),
                        task_text=user_msg,
                    )

                yield 'data: [DONE]\n\n'
                return
    except Exception as exc:
        yield sse_encode({'error': {'message': str(exc), 'code': 'internal_error'}})
        yield 'data: [DONE]\n\n'


def _sse_data_values(chunk):
    values = []
    for line in str(chunk).splitlines():
        if line.startswith('data: '):
            values.append(line[len('data: '):])
    return values


async def _collect_responses_completion(gen):
    """Drain the ``/v1/responses`` SSE generator into a single JSON body.

    Pre-flight error events (bad JSON / unknown previous_response_id /
    invalid resume input) are emitted with ``error`` but no ``object`` key,
    and arrive before the run starts — map them to HTTP 4xx mirroring the
    OpenAI error envelope. Terminal run frames (``completed`` / ``failed``
    / ``requires_action``) carry the full response object (``object="response"``)
    and are returned verbatim with HTTP 200, after stripping the SSE-only
    ``type`` and ``sequence_number`` keys ``response_sse`` injects.
    """
    final_payload = None
    async for chunk in gen:
        for raw in _sse_data_values(chunk):
            if raw == '[DONE]':
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if payload.get('object') != 'response' and payload.get('error'):
                err = payload.get('error') or {}
                code = err.get('code') or 'invalid_request_error'
                status = 500 if code in ('internal_error', 'run_failed') else 400
                if code in ('resume_not_found', 'response_not_found'):
                    status = 404
                return JSONResponse(
                    {'error': {
                        'message': err.get('message') or 'Response run failed',
                        'type': err.get('type') or 'invalid_request_error',
                        'code': code,
                    }},
                    status_code=status,
                )
            if payload.get('type') in (
                'response.completed', 'response.failed', 'response.requires_action',
            ):
                final_payload = payload
    if final_payload is None:
        return JSONResponse(
            {'error': {
                'message': 'Stream ended without a terminal response event',
                'type': 'internal_error', 'code': 'internal_error',
            }},
            status_code=500,
        )
    return {k: v for k, v in final_payload.items()
            if k not in ('type', 'sequence_number')}


async def _collect_chat_completion(gen, *, model):
    completion_id = ''
    created = int(time.time())
    content_parts = []
    tool_calls = []
    finish_reason = None
    usage = None
    async for chunk in gen:
        for raw in _sse_data_values(chunk):
            if raw == '[DONE]':
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if payload.get('error'):
                err = payload.get('error') or {}
                code = err.get('code') or 'invalid_request_error'
                status = 500 if code in ('internal_error', 'run_failed') else 400
                if code in ('resume_not_found', 'response_not_found'):
                    status = 404
                return JSONResponse(
                    {'error': {
                        'message': err.get('message') or 'Chat completion failed',
                        'type': err.get('type') or 'invalid_request_error',
                        'code': code,
                    }},
                    status_code=status,
                )
            completion_id = payload.get('id') or completion_id
            created = payload.get('created') or created
            usage = payload.get('usage') or usage
            for choice in payload.get('choices') or []:
                delta = choice.get('delta') or {}
                if delta.get('content'):
                    content_parts.append(delta.get('content') or '')
                if delta.get('tool_calls'):
                    tool_calls.extend(delta.get('tool_calls') or [])
                if choice.get('finish_reason'):
                    finish_reason = choice.get('finish_reason')
    message = {'role': 'assistant', 'content': ''.join(content_parts)}
    if tool_calls:
        message['tool_calls'] = tool_calls
        if not message['content']:
            message['content'] = None
    return {
        'id': completion_id or f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion',
        'created': created,
        'model': model,
        'choices': [{
            'index': 0,
            'message': message,
            'finish_reason': finish_reason or 'stop',
        }],
        'usage': usage,
    }


@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    if not isinstance(body, dict):
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    if request.headers.get('x-session-id'):
        return JSONResponse(openai_error(
            'X-Session-Id is no longer supported on Chat Completions; include conversation history in messages',
            code='unsupported_legacy_header',
        ), status_code=400)
    aliyun_credentials, error_response = _pop_aliyun_credentials_response(body)
    if error_response is not None:
        return error_response
    raw_model = body.get('model')
    model_override = None
    if raw_model:
        if not isinstance(raw_model, str):
            return JSONResponse(openai_error("'model' must be a string"), status_code=400)
        try:
            model_override = runtime_config.validate_model_name(raw_model)
        except ValueError as exc:
            return JSONResponse(openai_error(str(exc)), status_code=400)
    model = model_override or runtime_config.get_active_model()
    messages = body.get('messages')
    if not isinstance(messages, list):
        return JSONResponse(openai_error("'messages' must be an array"), status_code=400)
    cwd = body.get('cwd')
    stream = bool(body.get('stream', False))
    allow_hitl = bool(body.get('allow_hitl', bool(messages and isinstance(messages[-1], dict) and messages[-1].get('role') == 'tool')))
    aliyun_profile, error_response = _create_aliyun_profile_response(aliyun_credentials)
    if error_response is not None:
        return error_response
    gen = _sdk_chat_stream(
        body=body, model=model, cwd=cwd, allow_hitl=allow_hitl,
        tool_env=aliyun_profile.env if aliyun_profile is not None else None,
    )
    if stream:
        gen = _with_aliyun_profile_cleanup(gen, aliyun_profile)
        return StreamingResponse(_with_sse_keepalive(gen), media_type='text/event-stream',
                                 headers=stream_headers())
    try:
        return await _collect_chat_completion(gen, model=model)
    finally:
        _cleanup_aliyun_profile(aliyun_profile)


@app.get('/v1/sessions')
def list_sessions():
    return {'object': 'list', 'data': service.list_sessions(user_id=SERVER_USER_ID)}


@app.post('/v1/sessions')
async def create_session(request: Request):
    body = await request.json() if request.headers.get('content-length') not in (None, '0') else {}
    try:
        sess = service.create_session(user_id=SERVER_USER_ID, cwd=(body or {}).get('cwd'))
    except WorkspaceViolation as e:
        raise backend_error(e) from e
    loaded = service.store.load(sess.sid, user_id=SERVER_USER_ID) or {}
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
def get_session(session_id: str):
    loaded = service.store.load(session_id, user_id=SERVER_USER_ID)
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
        'pending_hitl': service.pending_hitl_for_session(session_id, user_id=SERVER_USER_ID),
    }
    if getattr(config, 'EXPOSE_SESSION_DEBUG', False):
        payload['llm_history'] = loaded.get('llm_history', [])
        payload['handler_state'] = loaded.get('handler_state')
    return payload


@app.post('/v1/sessions/{session_id}/regenerate')
async def regenerate_session_answer(session_id: str, request: Request):
    try:
        body = await request.json() if request.headers.get('content-length') not in (None, '0') else {}
    except Exception:
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    if body is None:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse(openai_error('Invalid JSON'), status_code=400)
    try:
        result = service.regenerate_session(session_id, user_id=SERVER_USER_ID)
    except (SessionBusyError, ServiceCapacityError, NoRegeneratableAnswerError, WorkspaceViolation, ToolWorkspaceViolation) as exc:
        raise backend_error(exc) from exc
    if result is None:
        raise HTTPException(status_code=404, detail='Session not found')

    cwd = body.get('cwd')
    raw_model = body.get('model')
    model_override = None
    if raw_model:
        if not isinstance(raw_model, str):
            return JSONResponse(openai_error("'model' must be a string"), status_code=400)
        try:
            model_override = runtime_config.validate_model_name(raw_model)
        except ValueError as exc:
            return JSONResponse(openai_error(str(exc)), status_code=400)
    model = model_override or runtime_config.get_active_model()

    sdk_body = {'input': result['user_text'], 'stream': True, 'cwd': cwd}
    gen = _sdk_response_stream(
        body=sdk_body, model=model, model_override=model_override,
        instructions='', conversation='',
        session_id=result['session_id'], cwd=cwd, store=True, previous_response_id='',
    )
    return StreamingResponse(_with_sse_keepalive(gen), media_type='text/event-stream',
                             headers=stream_headers())


@app.delete('/v1/sessions/{session_id}')
def delete_session(session_id: str):
    deleted = service.delete_session(session_id, user_id=SERVER_USER_ID)
    if not deleted:
        raise HTTPException(status_code=404, detail='Session not found')
    return {'deleted': True, 'session_id': session_id}


@app.post('/v1/sessions/{session_id}/cancel')
def cancel_session(session_id: str):
    cancelled = service.cancel_session(session_id, user_id=SERVER_USER_ID)
    if not cancelled:
        raise HTTPException(status_code=404, detail='Session not found')
    return {'cancelled': True, 'session_id': session_id}
