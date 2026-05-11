import datetime
import json
import os
import queue
import re
import sys
import threading
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent_events import agent_message_chunk, ask_user, done, stop_reason  # noqa: E402
from backend.background_review import schedule_background_memory_review  # noqa: E402
from backend.memory_scope import ensure_memory_scope, memory_scope_for, read_index  # noqa: E402
from backend.tool_schemas import main_tools_schema  # noqa: E402
from backend.workspace import WorkspaceManager, WorkspaceViolation  # noqa: E402
from agent_loop import StepOutcome, agent_runner_loop, sanitize_for_archive  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from session_store import SERVER_USER_ID, SessionStore  # noqa: E402
from skill_manager import (  # noqa: E402
    build_skill_user_input,
    get_skills_prompt,
    get_use_skill_schema,
    match_skill,
    scan_skills,
)
from tools import GenericHandler, WorkspaceViolation as ToolWorkspaceViolation  # noqa: E402
import settings as config  # noqa: E402


TOOLS_SCHEMA = main_tools_schema()
SYS_PROMPT_BASE = open(os.path.join(ROOT, 'prompts', 'sys_prompt.txt'), encoding='utf-8').read()
SKILLS = scan_skills(os.path.join(ROOT, 'skills'))
if SKILLS:
    TOOLS_SCHEMA.append(get_use_skill_schema())


SESSION_IDLE = 'idle'
SESSION_RUNNING = 'running'
SESSION_WAITING_USER = 'waiting_user'
SESSION_COMPLETED = 'completed'
SESSION_FAILED = 'failed'
SESSION_CANCELLED = 'cancelled'
ACTIVE_SESSION_STATUSES = {SESSION_RUNNING, SESSION_WAITING_USER}
FAILED_EXIT_RESULTS = {
    'MAX_TURNS_EXCEEDED',
    'ERROR',
    'WORKSPACE_VIOLATION',
}


class SessionBusyError(RuntimeError):
    def __init__(self, session_id, status):
        self.session_id = session_id
        self.status = status
        super().__init__(f'session_busy: session {session_id} is {status}')


class NoRegeneratableAnswerError(RuntimeError):
    def __init__(self, session_id):
        self.session_id = session_id
        super().__init__(f'no_regeneratable_answer: session {session_id} has no completed answer to regenerate')


class ServiceCapacityError(RuntimeError):
    def __init__(self, scope, limit):
        self.scope = scope
        self.limit = limit
        super().__init__(f'capacity_exceeded: {scope} active run limit {limit} reached')


def final_status_for_exit_reason(exit_reason):
    result = (exit_reason or {}).get('result')
    if result in FAILED_EXIT_RESULTS:
        return SESSION_FAILED
    return SESSION_COMPLETED


def exit_reason_error(exit_reason):
    exit_reason = exit_reason or {}
    if exit_reason.get('msg'):
        return exit_reason.get('msg', '')
    if exit_reason.get('result') == 'MAX_TURNS_EXCEEDED':
        return 'MAX_TURNS_EXCEEDED'
    return ''


def long_term_memory_enabled(user_id=SERVER_USER_ID):
    return (user_id or SERVER_USER_ID) == SERVER_USER_ID or getattr(config, 'ENABLE_LONG_TERM_MEMORY_FOR_USERS', False)


def handler_memory_scope(user_id=SERVER_USER_ID):
    user_id = user_id or SERVER_USER_ID
    if user_id == SERVER_USER_ID or long_term_memory_enabled(user_id):
        return memory_scope_for(ROOT, user_id)
    if getattr(config, 'ENABLE_SHARED_MEMORY_FOR_USERS', False):
        return memory_scope_for(ROOT, SERVER_USER_ID)
    return memory_scope_for(ROOT, user_id)


def build_system_prompt(user_id=SERVER_USER_ID):
    user_id = user_id or SERVER_USER_ID
    notice = ''
    if user_id == SERVER_USER_ID:
        scope = memory_scope_for(ROOT, user_id)
        idx = read_index(scope)
        notice = f'\n[MEMORY SCOPE] Service memory root: {scope.root}\n'
    elif long_term_memory_enabled(user_id):
        scope = memory_scope_for(ROOT, user_id)
        idx = read_index(scope)
        notice = f'\n[MEMORY SCOPE] User-private memory root: {scope.root}\n'
    elif getattr(config, 'ENABLE_SHARED_MEMORY_FOR_USERS', False):
        scope = memory_scope_for(ROOT, SERVER_USER_ID)
        idx = read_index(scope)
        notice = '\n[MEMORY SCOPE] Shared service memory is readable for this user; user memory updates are disabled.\n'
    else:
        idx = '(empty)'
        notice = '\n[MEMORY SCOPE] Long-term memory is disabled for this user.\n'
    return SYS_PROMPT_BASE + notice + '\n' + idx + get_skills_prompt(SKILLS)


def archive_session(client, task, exit_reason, user_id=SERVER_USER_ID):
    scope = memory_scope_for(ROOT, user_id)
    ensure_memory_scope(scope)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(scope.archive_dir, f'{ts}_{uuid.uuid4().hex[:8]}.md')
    archived_task = sanitize_for_archive(task)
    archived_exit = sanitize_for_archive(exit_reason)
    archived_history = sanitize_for_archive(client.history)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# Task ({ts})\n{archived_task}\n\n')
        f.write(f'## Exit\n```json\n{json.dumps(archived_exit, ensure_ascii=False, default=str, indent=2)}\n```\n\n')
        f.write('## History\n```json\n')
        json.dump(archived_history, f, ensure_ascii=False, default=str, indent=2)
        f.write('\n```\n')


def flatten_message_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get('type') == 'text':
                parts.append(block.get('text', ''))
        return '\n'.join(p for p in parts if p)
    return '' if content is None else str(content)


def last_user_text(messages):
    for msg in reversed(messages or []):
        if msg.get('role') == 'user':
            return flatten_message_content(msg.get('content'))
    return ''


class HttpHandler(GenericHandler):
    def __init__(
        self,
        cwd,
        mini_agent_root,
        display_q,
        ask_q,
        session,
        workspace_root=None,
        readonly_roots=None,
        writable_roots=None,
        memory_root=None,
        long_term_memory_enabled=True,
    ):
        super().__init__(
            cwd,
            mini_agent_root,
            workspace_root=workspace_root,
            readonly_roots=readonly_roots,
            writable_roots=writable_roots,
            memory_root=memory_root,
            long_term_memory_enabled=long_term_memory_enabled,
        )
        self._dq = display_q
        self._aq = ask_q
        self._session = session

    def do_ask_user(self, args, response):
        question = args.get('question', '请提供输入：')
        candidates = args.get('candidates') or []
        if getattr(self._session, 'output_mode', 'events') == 'text':
            text = f'\n[Agent asks] {question}\n'
            if candidates:
                text += ''.join(f'{i}. {c}\n' for i, c in enumerate(candidates, 1))
            self._session._on_text_chunk(text, check_cancel=False)
        else:
            self._session.emit_event(ask_user(question, candidates), check_cancel=False)
            self._session.emit_event(done('end_turn'), check_cancel=False)
        self._session.mark_waiting_for_user()
        self._session.turn_done_evt.set()
        answer = self._aq.get()
        if answer.strip().isdigit() and candidates and 1 <= int(answer) <= len(candidates):
            answer = candidates[int(answer) - 1]
        return StepOutcome({'status': 'answered', 'answer': answer},
                           next_prompt=f'用户回答：{answer}\n根据答案继续推进任务。')


class AgentSession:
    def __init__(self, service, sid, user_id=SERVER_USER_ID, cwd=None):
        self.service = service
        self.sid = sid
        self.user_id = user_id
        self.cwd, self.workspace_root, self.readonly_roots = self.service.prepare_workspace(user_id, sid, cwd)
        self.workspace_path = self.workspace_root or self.cwd
        self.memory_scope = handler_memory_scope(user_id)
        self.client = self._create_client()
        self.client.history_changed = self.save
        self.handler = None
        self.ui_msgs = []
        self.display_q = queue.Queue()
        self.ask_q = queue.Queue()
        self.turn_done_evt = threading.Event()
        self.cancel_evt = threading.Event()
        self.worker = None
        self.exit_reason = None
        self.output_mode = 'events'
        self.status = SESSION_IDLE
        self.active_run_id = ''
        self._lock = threading.RLock()

    def _create_client(self):
        return LLMClient(
            api_key=config.API_KEY,
            api_base=getattr(config, 'API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            model=getattr(config, 'MODEL', 'qwen-plus'),
            max_tokens=getattr(config, 'MAX_TOKENS', 8192),
            history_trim_tokens=getattr(config, 'HISTORY_TRIM_TOKENS', 80000),
            timeout=getattr(config, 'TIMEOUT', 300),
        )

    def restore_from(self, loaded):
        if not loaded:
            return
        self.client.history = loaded.get('llm_history', []) or []
        self.ui_msgs = loaded.get('ui_messages', []) or []
        loaded_status = loaded.get('status') or SESSION_IDLE
        self.status = SESSION_IDLE if loaded_status in ACTIVE_SESSION_STATUSES else loaded_status
        self.active_run_id = ''
        if loaded.get('workspace_path') and self.workspace_root:
            self.workspace_path = loaded.get('workspace_path')
            self.workspace_root = self.workspace_path
            self.cwd = self.workspace_path
            os.makedirs(self.workspace_path, exist_ok=True)
        state = loaded.get('handler_state') or {}
        self.handler = self._handler_from_state(state) if state else None

    def _handler_from_state(self, state):
        if not state:
            return None
        handler = self._new_handler()
        handler.history_info = list(state.get('history_info', []) or [])
        handler.working = dict(state.get('working', {}) or {})
        handler.todos = list(state.get('todos', []) or [])
        active_skill = handler.working.get('active_skill')
        if active_skill in SKILLS:
            handler.allow_readonly_root(os.path.dirname(SKILLS[active_skill].path))
        return handler

    def snapshot_handler_state(self):
        if self.handler is None:
            return None
        return {
            'history_info': list(self.handler.history_info),
            'working': dict(self.handler.working),
            'todos': list(getattr(self.handler, 'todos', []) or []),
        }

    def save(self):
        with self._lock:
            llm_history = list(self.client.history)
            ui_msgs = list(self.ui_msgs)
        self.service.store.save(
            session_id=self.sid,
            user_id=self.user_id,
            llm_history=llm_history,
            ui_messages=ui_msgs,
            handler_state=self.snapshot_handler_state(),
            status=self.status,
            active_run_id=self.active_run_id,
            workspace_path=self.workspace_path,
        )

    def _new_handler(self):
        allow_long_term = long_term_memory_enabled(self.user_id)
        writable_roots = [self.memory_scope.root] if self.workspace_root and allow_long_term else []
        readonly_roots = list(self.readonly_roots)
        if (
            self.workspace_root
            and self.user_id != SERVER_USER_ID
            and not allow_long_term
            and getattr(config, 'ENABLE_SHARED_MEMORY_FOR_USERS', False)
        ):
            readonly_roots.append(self.memory_scope.root)
        handler = HttpHandler(
            self.cwd,
            ROOT,
            self.display_q,
            self.ask_q,
            self,
            workspace_root=self.workspace_root,
            readonly_roots=readonly_roots,
            writable_roots=writable_roots,
            memory_root=self.memory_scope.root,
            long_term_memory_enabled=allow_long_term,
        )
        handler.cancel_evt = self.cancel_evt
        return handler

    def _append_ui_message_locked(self, role, content='', events=None):
        msg = {'role': role, 'content': content}
        if events:
            msg['events'] = list(events)
        self.ui_msgs.append(msg)
        return msg

    def append_ui_message(self, role, content='', events=None):
        with self._lock:
            self._append_ui_message_locked(role, content, events)
        self.save()

    def _ensure_assistant_message(self):
        with self._lock:
            if not self.ui_msgs or self.ui_msgs[-1].get('role') != 'assistant':
                self.ui_msgs.append({'role': 'assistant', 'content': '', 'events': []})
            else:
                self.ui_msgs[-1].setdefault('events', [])

    def _record_assistant_event(self, event):
        self._ensure_assistant_message()
        with self._lock:
            msg = self.ui_msgs[-1]
            if event.get('sessionUpdate') == 'agent_message_chunk':
                text = ((event.get('content') or {}).get('text') or '')
                msg['content'] = (msg.get('content') or '') + text
            else:
                msg.setdefault('events', []).append(event)
        self.save()

    def mark_waiting_for_user(self):
        with self._lock:
            self.status = SESSION_WAITING_USER
            run_id = self.active_run_id
        if run_id:
            self.service.store.set_run_status(run_id, self.user_id, SESSION_WAITING_USER)
        self.save()

    def run_or_answer(self, text, mode='events'):
        with self._lock:
            if self.worker is not None and self.worker.is_alive():
                if self.status == SESSION_WAITING_USER:
                    self._append_ui_message_locked('user', text)
                    self.status = SESSION_RUNNING
                    self.turn_done_evt.clear()
                    self.ask_q.put(text)
                    if self.active_run_id:
                        self.service.store.set_run_status(self.active_run_id, self.user_id, SESSION_RUNNING)
                    self.save()
                    return self.active_run_id
                raise SessionBusyError(self.sid, self.status)

            if self.worker is not None and not self.worker.is_alive():
                self.worker.join(timeout=0.1)
                self.worker = None

            return self._spawn_worker_locked(text, mode=mode)

    def _spawn_worker_locked(
        self,
        text,
        mode='events',
        append_ui=True,
        run_id=None,
        persist_run_record=True,
        pre_run_snapshot=None,
        check_capacity=True,
    ):
        if check_capacity:
            self.service.ensure_capacity(self.user_id, exclude_sid=self.sid)
        self.output_mode = mode
        pre_run_snapshot = pre_run_snapshot or {
            'llm_history': list(self.client.history),
            'handler_state': self.snapshot_handler_state(),
            'ui_message_count': len(self.ui_msgs),
            'workspace_path': self.workspace_path,
            'input_text': text,
        }
        sk, sk_args = match_skill(text, SKILLS)
        task_text = build_skill_user_input(sk, sk_args) if sk else text
        prev = self.handler
        handler = self._new_handler()
        if sk:
            handler.allow_readonly_root(os.path.dirname(sk.path))
            handler.working['active_skill'] = sk.name
            handler.working['related_sop'] = f'skills/{sk.name}/SKILL.md'
        if prev:
            handler.history_info = list(prev.history_info)
            handler.todos = [
                dict(todo)
                for todo in getattr(prev, 'todos', []) or []
                if todo.get('status') in ('pending', 'in_progress', 'blocked')
            ]
            if 'key_info' in prev.working:
                ki = re.sub(r'\n\[SYSTEM\] 此为.*?工作记忆[。\n]*', '', prev.working['key_info'])
                handler.working['key_info'] = ki
                ps = prev.working.get('passed_sessions', 0) + 1
                handler.working['passed_sessions'] = ps
                handler.working['key_info'] += (
                    f'\n[SYSTEM] 此为 {ps} 个对话前设置的key_info，'
                    f'若已在新任务，先更新或清除工作记忆。\n'
                )
        handler.history_info.append(f"[USER]: {task_text[:200]}")
        self.handler = handler

        user_input = task_text
        if prev:
            user_input = handler._anchor_prompt() + f'\n\n### 用户当前消息\n{task_text}'

        if append_ui:
            self._append_ui_message_locked('user', text)
        self._ensure_assistant_message()
        self.turn_done_evt.clear()
        self.cancel_evt.clear()
        self.exit_reason = None
        self.status = SESSION_RUNNING
        self.active_run_id = run_id or f'run_{uuid.uuid4().hex}'
        run_id = self.active_run_id
        if persist_run_record:
            self.service.store.create_run_record(
                self.sid,
                self.user_id,
                run_id,
                mode,
                status=SESSION_RUNNING,
                metadata={'pre_run_snapshot': pre_run_snapshot},
            )
        self.save()
        self.worker = threading.Thread(target=self._run_loop, args=(user_input, task_text, mode, run_id), daemon=True)
        self.worker.start()
        return run_id

    def regenerate_last_answer(self, mode='events'):
        with self._lock:
            if self.worker is not None and self.worker.is_alive():
                raise SessionBusyError(self.sid, self.status)
            if self.worker is not None and not self.worker.is_alive():
                self.worker.join(timeout=0.1)
                self.worker = None

            run_id = f'run_{uuid.uuid4().hex}'
            result = self.service.store.try_start_regenerate_run(
                session_id=self.sid,
                user_id=self.user_id,
                run_id=run_id,
                mode=mode,
                workspace_path=self.workspace_path,
                max_global_runs=int(getattr(config, 'MAX_GLOBAL_RUNS', 0) or 0),
                max_user_runs=int(getattr(config, 'MAX_USER_RUNS', 0) or 0),
            )
            if result['status'] == 'not_found':
                return None
            if result['status'] == 'busy':
                raise SessionBusyError(self.sid, result.get('session_status', 'running'))
            if result['status'] == 'capacity':
                raise ServiceCapacityError(result.get('scope', 'global'), result.get('limit', 0))
            if result['status'] == 'no_regeneratable_answer':
                raise NoRegeneratableAnswerError(self.sid)

            self.client.history = list(result.get('llm_history') or [])
            self.ui_msgs = list(result.get('ui_messages') or [])
            self.workspace_path = result.get('workspace_path') or self.workspace_path
            if self.workspace_root:
                self.workspace_root = self.workspace_path
                self.cwd = self.workspace_path
                os.makedirs(self.workspace_path, exist_ok=True)
            self.handler = self._handler_from_state(result.get('handler_state')) if result.get('handler_state') else None
            started_run_id = self._spawn_worker_locked(
                result['input_text'],
                mode=mode,
                append_ui=False,
                run_id=run_id,
                persist_run_record=False,
                pre_run_snapshot={
                    'llm_history': list(result.get('llm_history') or []),
                    'handler_state': result.get('handler_state'),
                    'ui_message_count': max(0, len(result.get('ui_messages') or []) - 2),
                    'workspace_path': result.get('workspace_path') or self.workspace_path,
                    'input_text': result['input_text'],
                },
                check_capacity=False,
            )
            return {
                'session_id': self.sid,
                'run_id': started_run_id,
                'stream_from': '0-0',
                'regenerated_from_run_id': result.get('regenerated_from_run_id') or '',
            }

    def _run_loop(self, user_input, task_text, mode='events', run_id=''):
        final_status = SESSION_COMPLETED
        try:
            kwargs = {
                'client': self.client,
                'system_prompt': build_system_prompt(self.user_id),
                'user_input': user_input,
                'handler': self.handler,
                'tools_schema': TOOLS_SCHEMA,
                'max_turns': getattr(config, 'MAX_TURNS', 40),
            }
            if mode == 'text':
                kwargs['on_chunk'] = self._on_text_chunk
            else:
                kwargs['on_event'] = self._on_event
            self.exit_reason = agent_runner_loop(**kwargs)
            final_status = final_status_for_exit_reason(self.exit_reason)
        except KeyboardInterrupt:
            self.exit_reason = {'result': 'INTERRUPTED'}
            final_status = SESSION_CANCELLED
            if mode == 'text':
                pass
            else:
                self.emit_event(done('cancelled'), check_cancel=False)
        except (WorkspaceViolation, ToolWorkspaceViolation) as e:
            self.exit_reason = {'result': 'WORKSPACE_VIOLATION', 'msg': str(e)}
            final_status = SESSION_FAILED
            if mode == 'text':
                self._on_text_chunk(f'\n**[Workspace violation]** {e}\n', check_cancel=False)
            else:
                self.emit_event(agent_message_chunk(f'**[Workspace violation]** {e}'), check_cancel=False)
                self.emit_event(done(stop_reason(self.exit_reason)), check_cancel=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.exit_reason = {'result': 'ERROR', 'msg': str(e)}
            final_status = SESSION_FAILED
            if mode == 'text':
                self._on_text_chunk(f'\n**[Error]** {e}\n', check_cancel=False)
            else:
                self.emit_event(agent_message_chunk(f'**[Error]** {e}'), check_cancel=False)
                self.emit_event(done(stop_reason(self.exit_reason)), check_cancel=False)
        finally:
            review_history = list(getattr(self.client, 'history', []) or [])
            review_active_skill = ''
            handler = getattr(self, 'handler', None)
            working = getattr(handler, 'working', {}) or {}
            review_active_skill = working.get('active_skill') or ''
            memory_scope = getattr(self, 'memory_scope', None)
            review_memory_root = getattr(memory_scope, 'root', '') or ''
            review_long_term_enabled = long_term_memory_enabled(self.user_id)
            try:
                archive_session(self.client, task_text, self.exit_reason, user_id=self.user_id)
            except Exception:
                pass
            should_finish_run = False
            with self._lock:
                if self.active_run_id == run_id and self.status != SESSION_WAITING_USER:
                    if self.status == SESSION_CANCELLED:
                        final_status = SESSION_CANCELLED
                    self.status = final_status
                    self.active_run_id = ''
                    self.worker = None
                    should_finish_run = True
            self.save()
            if should_finish_run:
                self.service.store.finish_run(
                    self.sid,
                    self.user_id,
                    run_id,
                    final_status,
                    error=exit_reason_error(self.exit_reason),
                )
            self.turn_done_evt.set()
            if should_finish_run and final_status == SESSION_COMPLETED:
                try:
                    schedule_background_memory_review(
                        user_id=self.user_id,
                        session_id=self.sid,
                        run_id=run_id,
                        task_text=task_text,
                        llm_history=review_history,
                        memory_root=review_memory_root,
                        active_skill=review_active_skill,
                        final_status=final_status,
                        long_term_enabled=review_long_term_enabled,
                        use_celery=False,
                    )
                except Exception:
                    pass

    def emit_event(self, event, check_cancel=True):
        if check_cancel and self.cancel_evt.is_set():
            raise KeyboardInterrupt('User cancelled')
        self.display_q.put({'event': event})
        self._record_assistant_event(event)

    def _on_event(self, event):
        self.emit_event(event)

    def _on_text_chunk(self, chunk, check_cancel=True):
        if check_cancel and self.cancel_evt.is_set():
            raise KeyboardInterrupt('User cancelled')
        if not chunk:
            return
        self.display_q.put({'text': chunk})
        self._ensure_assistant_message()
        with self._lock:
            msg = self.ui_msgs[-1]
            msg['content'] = (msg.get('content') or '') + chunk
        self.save()

    def cancel(self):
        with self._lock:
            if self.status in ACTIVE_SESSION_STATUSES:
                self.status = SESSION_CANCELLED
            self.cancel_evt.set()
        try:
            self.ask_q.put_nowait('[Cancelled]')
        except queue.Full:
            pass
        self.save()

    def iter_events(self):
        while True:
            try:
                item = self.display_q.get(timeout=0.1)
            except queue.Empty:
                if self.turn_done_evt.is_set():
                    break
                continue
            if 'event' in item:
                yield item['event']
            if self.turn_done_evt.is_set() and self.display_q.empty():
                break

    def iter_text(self):
        while True:
            try:
                item = self.display_q.get(timeout=0.1)
            except queue.Empty:
                if self.turn_done_evt.is_set():
                    break
                continue
            if 'text' in item:
                yield item['text']
            elif 'event' in item and item['event'].get('sessionUpdate') == 'agent_message_chunk':
                yield ((item['event'].get('content') or {}).get('text') or '')
            if self.turn_done_evt.is_set() and self.display_q.empty():
                break

    def is_running(self):
        return self.status in ACTIVE_SESSION_STATUSES or (self.worker is not None and self.worker.is_alive())


class AgentService:
    def __init__(self):
        self.store = SessionStore(os.path.join(ROOT, 'memory', 'sessions'))
        workspace_root = getattr(config, 'WORKSPACE_ROOT', os.path.join(ROOT, 'workspaces'))
        self.workspace_manager = WorkspaceManager(workspace_root)
        self._sessions = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(user_id, sid):
        return (user_id or SERVER_USER_ID, sid)

    def prepare_workspace(self, user_id, sid, cwd=None):
        user_id = user_id or SERVER_USER_ID
        enforce_server_workspace = getattr(config, 'ENFORCE_WORKSPACE_FOR_SERVER', False)
        if user_id == SERVER_USER_ID and not enforce_server_workspace:
            return os.path.abspath(cwd or ROOT), None, []
        workspace_cwd = self.workspace_manager.prepare_session(user_id, sid, cwd=cwd)
        return workspace_cwd, self.workspace_manager.session_root(user_id, sid), []

    def ensure_capacity(self, user_id, exclude_sid=None):
        max_global = int(getattr(config, 'MAX_GLOBAL_RUNS', 0) or 0)
        max_user = int(getattr(config, 'MAX_USER_RUNS', 0) or 0)
        if not max_global and not max_user:
            return
        with self._lock:
            active = [
                sess for (owner_id, sid), sess in self._sessions.items()
                if sid != exclude_sid and sess.status in ACTIVE_SESSION_STATUSES
            ]
            if max_global and len(active) >= max_global:
                raise ServiceCapacityError('global', max_global)
            if max_user:
                user_active = [sess for sess in active if sess.user_id == user_id]
                if len(user_active) >= max_user:
                    raise ServiceCapacityError('user', max_user)

    def create_session(self, user_id=SERVER_USER_ID, cwd=None):
        sid = str(uuid.uuid4())
        user_id = user_id or SERVER_USER_ID
        sess = AgentSession(self, sid, user_id=user_id, cwd=cwd)
        with self._lock:
            self._sessions[self._key(user_id, sid)] = sess
        sess.save()
        return sess

    def get_session(self, sid=None, user_id=SERVER_USER_ID, cwd=None):
        user_id = user_id or SERVER_USER_ID
        if not sid:
            return self.create_session(user_id=user_id, cwd=cwd)
        key = self._key(user_id, sid)
        with self._lock:
            sess = self._sessions.get(key)
            if sess:
                return sess
            sess = AgentSession(self, sid, user_id=user_id, cwd=cwd)
            loaded = self.store.load(sid, user_id=user_id)
            if loaded is not None:
                sess.restore_from(loaded)
            elif self.store.session_exists(sid):
                return None
            self._sessions[key] = sess
            if loaded is None:
                sess.save()
            return sess

    def load_session(self, sid, user_id=SERVER_USER_ID):
        user_id = user_id or SERVER_USER_ID
        key = self._key(user_id, sid)
        with self._lock:
            sess = self._sessions.get(key)
        loaded = self.store.load(sid, user_id=user_id)
        if loaded is None:
            return None
        if sess is None:
            sess = AgentSession(self, sid, user_id=user_id)
            sess.restore_from(loaded)
            with self._lock:
                self._sessions[key] = sess
        return sess

    def list_sessions(self, user_id=SERVER_USER_ID):
        user_id = user_id or SERVER_USER_ID
        rows = self.store.list_sessions(user_id=user_id)
        with self._lock:
            running = {
                sid: sess.is_running()
                for (owner_id, sid), sess in self._sessions.items()
                if owner_id == user_id
            }
        for row in rows:
            row['running'] = running.get(row['session_id'], False) or row.get('status') in ACTIVE_SESSION_STATUSES
        return rows

    def delete_session(self, sid, user_id=SERVER_USER_ID):
        user_id = user_id or SERVER_USER_ID
        key = self._key(user_id, sid)
        with self._lock:
            sess = self._sessions.pop(key, None)
        if sess:
            sess.cancel()
        return self.store.delete(sid, user_id=user_id)

    def cancel_session(self, sid, user_id=SERVER_USER_ID):
        user_id = user_id or SERVER_USER_ID
        with self._lock:
            sess = self._sessions.get(self._key(user_id, sid))
        if not sess:
            return False
        sess.cancel()
        return True

    def chat_text(self, sid, messages, user_id=SERVER_USER_ID, cwd=None):
        sess = self.get_session(sid, user_id=user_id, cwd=cwd)
        if sess is None:
            return None, iter(())
        text = last_user_text(messages)
        sess.run_or_answer(text, mode='text')
        return sess, sess.iter_text()

    def chat_events(self, sid, text, user_id=SERVER_USER_ID, cwd=None):
        sess = self.get_session(sid, user_id=user_id, cwd=cwd)
        if sess is None:
            return None, iter(())
        sess.run_or_answer(text, mode='events')
        return sess, sess.iter_events()

    def regenerate_session(self, sid, user_id=SERVER_USER_ID):
        sess = self.load_session(sid, user_id=user_id)
        if sess is None:
            return None
        return sess.regenerate_last_answer(mode='events')


def openai_chat_completion(model, content, session_id):
    now = int(time.time())
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion',
        'created': now,
        'model': model,
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': content},
            'finish_reason': 'stop',
        }],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        'metadata': {'session_id': session_id},
    }


def openai_chat_chunk(model, content):
    now = int(time.time())
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion.chunk',
        'created': now,
        'model': model,
        'choices': [{
            'index': 0,
            'delta': {'content': content},
            'finish_reason': None,
        }],
    }


def openai_role_chunk(model):
    now = int(time.time())
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion.chunk',
        'created': now,
        'model': model,
        'choices': [{
            'index': 0,
            'delta': {'role': 'assistant'},
            'finish_reason': None,
        }],
    }


def openai_done_chunk(model):
    now = int(time.time())
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion.chunk',
        'created': now,
        'model': model,
        'choices': [{
            'index': 0,
            'delta': {},
            'finish_reason': 'stop',
        }],
    }
