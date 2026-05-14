"""Celery worker entrypoint for executing agent runs outside FastAPI."""
import os
import re
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent_events import agent_message_chunk, ask_user, done, stop_reason  # noqa: E402
from agent_loop import StepOutcome, agent_runner_loop  # noqa: E402
from backend.background_review import schedule_background_memory_review  # noqa: E402
from backend.agent_service import (  # noqa: E402
    SESSION_CANCELLED,
    SESSION_COMPLETED,
    SESSION_FAILED,
    SESSION_RUNNING,
    SESSION_WAITING_USER,
    archive_session,
    build_system_prompt,
    exit_reason_error,
    final_status_for_exit_reason,
    handler_memory_scope,
    long_term_memory_enabled,
)
from backend.celery_app import celery_app  # noqa: E402
from backend.redis_bus import RedisBus  # noqa: E402
from backend.tool_schemas import main_tools_schema  # noqa: E402
from backend.workspace import WorkspaceManager, WorkspaceViolation  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from session_store import SERVER_USER_ID, SessionStore  # noqa: E402
from skill_manager import build_skill_user_input, match_skill, scan_skills  # noqa: E402
from tools import GenericHandler, WorkspaceViolation as ToolWorkspaceViolation  # noqa: E402
import settings as config  # noqa: E402
import runtime_config  # noqa: E402


SKILLS = scan_skills(os.path.join(ROOT, 'skills'))


class RedisCancelEvent:
    def __init__(self, bus, run_id):
        self.bus = bus
        self.run_id = run_id

    def is_set(self):
        return self.bus.is_cancelled(self.run_id)


class CeleryHandler(GenericHandler):
    def __init__(
        self,
        cwd,
        mini_agent_root,
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
        self._session = session
        self.cancel_evt = session.cancel_evt

    def do_ask_user(self, args, response):
        question = args.get('question', '请提供输入：')
        candidates = args.get('candidates') or []
        self._session.emit_event(ask_user(question, candidates), check_cancel=False)
        self._session.mark_waiting_for_user()
        self._session.emit_event(done('end_turn'), check_cancel=False)
        answer = self._session.wait_for_answer()
        if answer.strip().isdigit() and candidates and 1 <= int(answer) <= len(candidates):
            answer = candidates[int(answer) - 1]
        return StepOutcome({'status': 'answered', 'answer': answer},
                           next_prompt=f'用户回答：{answer}\n根据答案继续推进任务。')


class WorkerSession:
    def __init__(self, run_id, session_id, user_id, task_text, mode='events', cwd=None):
        self.run_id = run_id
        self.sid = session_id
        self.user_id = user_id or SERVER_USER_ID
        self.task_text = task_text
        self.mode = mode
        self.store = SessionStore(os.path.join(ROOT, 'memory', 'sessions'))
        self.bus = RedisBus()
        self.cancel_evt = RedisCancelEvent(self.bus, run_id)
        self.status = SESSION_RUNNING
        self.active_run_id = run_id
        # Layer 4 snapshot debounce: token chunks no longer trigger an
        # immediate SQLite write; logical-step events (tool calls, ask_user,
        # done) still flush eagerly so durability matches step boundaries.
        self._save_lock = threading.Lock()
        self._save_dirty = False
        self._save_last_flush_ms = 0.0
        self._save_timer = None
        self._save_interval_ms = max(0, int(getattr(config, 'SNAPSHOT_FLUSH_INTERVAL_MS', 750)))
        self.workspace_manager = WorkspaceManager(getattr(config, 'WORKSPACE_ROOT', os.path.join(ROOT, 'workspaces')))
        self.loaded = self.store.load(session_id, user_id=self.user_id)
        if self.loaded is None:
            raise RuntimeError(f'Session not found: {session_id}')
        self.cwd, self.workspace_root, self.readonly_roots = self._resolve_workspace(cwd)
        self.workspace_path = self.workspace_root or self.cwd
        self.memory_scope = handler_memory_scope(self.user_id)
        self.ui_msgs = self.loaded.get('ui_messages', []) or []
        self.client = self._create_client()
        self.client.history = self.loaded.get('llm_history', []) or []
        self.client.history_changed = self.save
        self.handler = self._build_handler()
        self.store.set_run_status(self.run_id, self.user_id, SESSION_RUNNING)

    def _resolve_workspace(self, cwd=None):
        if self.user_id == SERVER_USER_ID and not getattr(config, 'ENFORCE_WORKSPACE_FOR_SERVER', False):
            return os.path.abspath(cwd or self.loaded.get('workspace_path') or ROOT), None, []
        root = self.loaded.get('workspace_path') or self.workspace_manager.session_root(self.user_id, self.sid)
        os.makedirs(root, exist_ok=True)
        if cwd:
            cwd = cwd if os.path.isabs(str(cwd)) else os.path.join(root, str(cwd))
            cwd = os.path.abspath(cwd)
            try:
                in_workspace = os.path.commonpath([cwd, root]) == os.path.abspath(root)
            except ValueError:
                in_workspace = False
            if not in_workspace:
                raise WorkspaceViolation('workspace_violation: cwd escapes session workspace')
            os.makedirs(cwd, exist_ok=True)
            return cwd, root, []
        return root, root, []

    def _create_client(self):
        return LLMClient(
            api_key=config.API_KEY,
            api_base=getattr(config, 'API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            model=runtime_config.get_active_model(),
            max_tokens=getattr(config, 'MAX_TOKENS', 8192),
            history_trim_tokens=getattr(config, 'HISTORY_TRIM_TOKENS', 80000),
            timeout=getattr(config, 'TIMEOUT', 300),
        )

    def _build_handler(self):
        sk, sk_args = match_skill(self.task_text, SKILLS)
        task_text = build_skill_user_input(sk, sk_args) if sk else self.task_text
        state = self.loaded.get('handler_state') or {}
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
        handler = CeleryHandler(
            self.cwd,
            ROOT,
            self,
            workspace_root=self.workspace_root,
            readonly_roots=readonly_roots,
            writable_roots=writable_roots,
            memory_root=self.memory_scope.root,
            long_term_memory_enabled=allow_long_term,
        )
        handler.history_info = list(state.get('history_info', []) or [])
        handler.working = dict(state.get('working', {}) or {})
        handler.todos = list(state.get('todos', []) or [])
        active_skill = handler.working.get('active_skill')
        if active_skill in SKILLS:
            handler.allow_readonly_root(os.path.dirname(SKILLS[active_skill].path))
        if sk:
            handler.allow_readonly_root(os.path.dirname(sk.path))
            handler.working['active_skill'] = sk.name
            handler.working['related_sop'] = f'skills/{sk.name}/SKILL.md'
        if 'key_info' in handler.working:
            key_info = re.sub(r'\n\[SYSTEM\] 此为.*?工作记忆[。\n]*', '', handler.working['key_info'])
            passed = handler.working.get('passed_sessions', 0) + 1
            handler.working['key_info'] = (
                key_info
                + f'\n[SYSTEM] 此为 {passed} 个对话前设置的key_info，若已在新任务，先更新或清除工作记忆。\n'
            )
            handler.working['passed_sessions'] = passed
        handler.history_info.append(f"[USER]: {task_text[:200]}")
        self.user_input = task_text
        if state:
            self.user_input = handler._anchor_prompt() + f'\n\n### 用户当前消息\n{task_text}'
        return handler

    def snapshot_handler_state(self):
        return {
            'history_info': list(self.handler.history_info),
            'working': dict(self.handler.working),
            'todos': list(getattr(self.handler, 'todos', []) or []),
        }

    def save(self):
        with self._save_lock:
            self._cancel_pending_flush_locked()
            self._flush_locked()

    def _flush_locked(self):
        self.store.save_run_snapshot(
            session_id=self.sid,
            user_id=self.user_id,
            run_id=self.run_id,
            llm_history=list(self.client.history),
            ui_messages=list(self.ui_msgs),
            handler_state=self.snapshot_handler_state(),
            status=self.status,
            active_run_id=self.active_run_id,
            workspace_path=self.workspace_path,
        )
        self._save_dirty = False
        self._save_last_flush_ms = time.monotonic() * 1000.0

    def _cancel_pending_flush_locked(self):
        timer = self._save_timer
        if timer is not None:
            timer.cancel()
            self._save_timer = None

    def _timer_flush(self):
        with self._save_lock:
            self._save_timer = None
            if not self._save_dirty:
                return
            try:
                self._flush_locked()
            except Exception as e:
                # Timer thread runs outside the agent loop; swallow rather
                # than die silently and lose the next debounced flush.
                print(f'[WorkerSession] debounced flush failed: {e}', file=sys.stderr)

    def _schedule_save(self, force=False):
        # Force = step boundary (tool call, ask_user, finish, error). Token
        # chunks come through with force=False and may coalesce.
        with self._save_lock:
            if force or self._save_interval_ms <= 0:
                self._cancel_pending_flush_locked()
                self._flush_locked()
                return
            self._save_dirty = True
            now_ms = time.monotonic() * 1000.0
            if now_ms - self._save_last_flush_ms >= self._save_interval_ms:
                self._cancel_pending_flush_locked()
                self._flush_locked()
                return
            if self._save_timer is None:
                delay_s = self._save_interval_ms / 1000.0
                timer = threading.Timer(delay_s, self._timer_flush)
                timer.daemon = True
                self._save_timer = timer
                timer.start()

    def _ensure_assistant_message(self):
        if not self.ui_msgs or self.ui_msgs[-1].get('role') != 'assistant':
            self.ui_msgs.append({'role': 'assistant', 'content': '', 'events': []})
        else:
            self.ui_msgs[-1].setdefault('events', [])

    def _record_assistant_event(self, event):
        self._ensure_assistant_message()
        msg = self.ui_msgs[-1]
        update = event.get('sessionUpdate')
        if update == 'agent_message_chunk':
            text = ((event.get('content') or {}).get('text') or '')
            msg['content'] = (msg.get('content') or '') + text
            self._schedule_save(force=False)
        else:
            msg.setdefault('events', []).append(event)
            self._schedule_save(force=True)

    def _check_cancel(self):
        if self.bus.is_cancelled(self.run_id):
            raise KeyboardInterrupt('User cancelled')

    def emit_event(self, event, check_cancel=True):
        if check_cancel:
            self._check_cancel()
        self.bus.publish_event(self.run_id, event)
        self._record_assistant_event(event)

    def _on_event(self, event):
        self.emit_event(event)

    def _on_text_chunk(self, chunk):
        if not chunk:
            return
        self.emit_event(agent_message_chunk(chunk))

    def mark_waiting_for_user(self):
        self.status = SESSION_WAITING_USER
        self.store.mark_waiting_user(self.sid, self.user_id, self.run_id)
        self.save()

    def wait_for_answer(self):
        answer = self.bus.wait_answer(self.run_id)
        self.ui_msgs.append({'role': 'user', 'content': answer})
        self.status = SESSION_RUNNING
        self.save()
        return answer

    def run(self):
        final_status = SESSION_COMPLETED
        exit_reason = {}
        completed_normally = False
        try:
            kwargs = {
                'client': self.client,
                'system_prompt': build_system_prompt(self.user_id),
                'user_input': self.user_input,
                'handler': self.handler,
                'tools_schema': json_tools_schema(),
                'max_turns': getattr(config, 'MAX_TURNS', 40),
            }
            if self.mode == 'text':
                kwargs['on_chunk'] = self._on_text_chunk
            else:
                kwargs['on_event'] = self._on_event
            exit_reason = agent_runner_loop(**kwargs)
            final_status = final_status_for_exit_reason(exit_reason)
            completed_normally = True
        except KeyboardInterrupt:
            final_status = SESSION_CANCELLED
            exit_reason = {'result': 'INTERRUPTED'}
            # Layer 3 single-writer: server only sets the Redis cancel flag;
            # this worker process owns every SQLite write for the run, so
            # record `cancel_requested_at` here instead of from FastAPI.
            try:
                self.store.request_cancel(self.sid, self.user_id, self.run_id)
            except Exception:
                pass
            self.emit_event(done('cancelled'), check_cancel=False)
        except (WorkspaceViolation, ToolWorkspaceViolation) as e:
            final_status = SESSION_FAILED
            exit_reason = {'result': 'WORKSPACE_VIOLATION', 'msg': str(e)}
            self.emit_event(agent_message_chunk(f'**[Workspace violation]** {e}'), check_cancel=False)
            self.emit_event(done(stop_reason(exit_reason)), check_cancel=False)
        except Exception as e:
            final_status = SESSION_FAILED
            exit_reason = {'result': 'ERROR', 'msg': str(e)}
            self.emit_event(agent_message_chunk(f'**[Error]** {e}'), check_cancel=False)
            self.emit_event(done(stop_reason(exit_reason)), check_cancel=False)
            raise
        finally:
            review_history = list(getattr(self.client, 'history', []) or [])
            review_active_skill = (getattr(self.handler, 'working', {}) or {}).get('active_skill') or ''
            if self.mode == 'text' and completed_normally:
                self.emit_event(done(stop_reason(exit_reason)), check_cancel=False)
            try:
                archive_session(self.client, self.task_text, exit_reason, user_id=self.user_id)
            except Exception:
                pass
            self.status = final_status
            self.active_run_id = ''
            self.save()
            self.store.finish_run(self.sid, self.user_id, self.run_id, final_status, error=exit_reason_error(exit_reason))
            if final_status == SESSION_COMPLETED:
                try:
                    schedule_background_memory_review(
                        user_id=self.user_id,
                        session_id=self.sid,
                        run_id=self.run_id,
                        task_text=self.task_text,
                        llm_history=review_history,
                        memory_root=self.memory_scope.root,
                        active_skill=review_active_skill,
                        final_status=final_status,
                        long_term_enabled=long_term_memory_enabled(self.user_id),
                        use_celery=True,
                    )
                except Exception:
                    pass


def json_tools_schema():
    tools_schema = main_tools_schema()
    if SKILLS:
        from skill_manager import get_use_skill_schema

        tools_schema.append(get_use_skill_schema())
    return tools_schema


@celery_app.task(name='pai_rag.run_agent')
def run_agent_task(run_id, session_id, user_id, text, mode='events', cwd=None):
    session = WorkerSession(run_id, session_id, user_id, text, mode=mode, cwd=cwd)
    session.run()
