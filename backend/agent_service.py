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

from agent_loop import StepOutcome, agent_runner_loop  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from session_store import SessionStore  # noqa: E402
from skill_manager import (  # noqa: E402
    build_skill_user_input,
    get_skills_prompt,
    get_use_skill_schema,
    match_skill,
    scan_skills,
)
from tools import GenericHandler, SEDIMENT_HOOK  # noqa: E402

try:
    import config
except ImportError as e:
    raise RuntimeError('config.py not found. Copy config_template.py to config.py first.') from e


TOOLS_SCHEMA = json.load(open(os.path.join(ROOT, 'tools_schema.json'), encoding='utf-8'))
SYS_PROMPT_BASE = open(os.path.join(ROOT, 'prompts', 'sys_prompt.txt'), encoding='utf-8').read()
SKILLS = scan_skills(os.path.join(ROOT, 'skills'))
if SKILLS:
    TOOLS_SCHEMA.append(get_use_skill_schema())


def build_system_prompt():
    idx_path = os.path.join(ROOT, 'memory', 'global_index.txt')
    idx = open(idx_path, encoding='utf-8').read() if os.path.exists(idx_path) else '(empty)'
    return SYS_PROMPT_BASE + '\n' + idx + get_skills_prompt(SKILLS)


def archive_session(client, task, exit_reason):
    archive_dir = os.path.join(ROOT, 'memory', 'L4_raw_sessions')
    os.makedirs(archive_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(archive_dir, f'{ts}.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# Task ({ts})\n{task}\n\n')
        f.write(f'## Exit\n```json\n{json.dumps(exit_reason, ensure_ascii=False, default=str, indent=2)}\n```\n\n')
        f.write('## History\n```json\n')
        json.dump(client.history, f, ensure_ascii=False, default=str, indent=2)
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


def encode_ask_user_block(ask):
    payload = json.dumps(ask, ensure_ascii=False)
    return f'\n[[ASK_USER]]{payload}[[/ASK_USER]]\n'


class HttpHandler(GenericHandler):
    def __init__(self, cwd, mini_agent_root, display_q, ask_q, session):
        super().__init__(cwd, mini_agent_root)
        self._dq = display_q
        self._aq = ask_q
        self._session = session

    def do_ask_user(self, args, response):
        question = args.get('question', '请提供输入：')
        candidates = args.get('candidates') or []
        ask = {'question': question, 'candidates': candidates}
        self._session.commit_response_so_far()
        self._session.append_ui_message('assistant', encode_ask_user_block(ask))
        self._dq.put({'ask': ask})
        self._session.turn_done_evt.set()
        answer = self._aq.get()
        if answer.strip().isdigit() and candidates and 1 <= int(answer) <= len(candidates):
            answer = candidates[int(answer) - 1]
        return StepOutcome({'status': 'answered', 'answer': answer},
                           next_prompt=f'用户回答：{answer}\n根据答案继续推进任务。')


class AgentSession:
    def __init__(self, service, sid, cwd=None):
        self.service = service
        self.sid = sid
        self.cwd = cwd or ROOT
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
        self.response_chunks = []
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
        state = loaded.get('handler_state') or {}
        if state:
            handler = HttpHandler(self.cwd, ROOT, self.display_q, self.ask_q, self)
            handler.history_info = state.get('history_info', [])
            handler.working = state.get('working', {})
            self.handler = handler

    def snapshot_handler_state(self):
        if self.handler is None:
            return None
        return {
            'history_info': list(self.handler.history_info),
            'working': dict(self.handler.working),
        }

    def save(self):
        with self._lock:
            llm_history = list(self.client.history)
            ui_msgs = list(self.ui_msgs)
        self.service.store.save(
            session_id=self.sid,
            llm_history=llm_history,
            ui_messages=ui_msgs,
            handler_state=self.snapshot_handler_state(),
        )

    def append_ui_message(self, role, content):
        with self._lock:
            msg = {'role': role, 'content': content}
            if (
                role == 'assistant'
                and self.ui_msgs
                and self.ui_msgs[-1].get('role') == 'assistant'
                and content.startswith(self.ui_msgs[-1].get('content') or '')
            ):
                self.ui_msgs[-1] = msg
            elif not (
                self.ui_msgs
                and self.ui_msgs[-1].get('role') == role
                and self.ui_msgs[-1].get('content') == content
            ):
                self.ui_msgs.append(msg)
        self.save()

    def commit_response_so_far(self):
        with self._lock:
            content = ''.join(self.response_chunks)
            self.response_chunks = []
        if content:
            self.append_ui_message('assistant', content)

    def run_or_answer(self, text):
        if self.worker is not None and self.worker.is_alive():
            if self.turn_done_evt.is_set() and self.exit_reason is not None:
                self.worker.join(timeout=0.1)
            if self.worker is not None and not self.worker.is_alive():
                self._spawn_worker(text)
                return
            self.append_ui_message('user', text)
            self.turn_done_evt.clear()
            self.ask_q.put(text)
            return
        self._spawn_worker(text)

    def _spawn_worker(self, text):
        sk, sk_args = match_skill(text, SKILLS)
        task_text = build_skill_user_input(sk, sk_args) if sk else text
        prev = self.handler
        handler = HttpHandler(self.cwd, ROOT, self.display_q, self.ask_q, self)
        if sk:
            handler.working['active_skill'] = sk.name
            handler.working['related_sop'] = f'skills/{sk.name}/SKILL.md'
        if prev:
            handler.history_info = list(prev.history_info)
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
        handler._done_hooks.append(SEDIMENT_HOOK)
        self.handler = handler

        user_input = task_text
        if prev:
            user_input = handler._anchor_prompt() + f'\n\n### 用户当前消息\n{task_text}'

        self.append_ui_message('user', text)
        self.turn_done_evt.clear()
        self.cancel_evt.clear()
        self.exit_reason = None
        with self._lock:
            self.response_chunks = []
        self.worker = threading.Thread(target=self._run_loop, args=(user_input, task_text), daemon=True)
        self.worker.start()

    def _run_loop(self, user_input, task_text):
        try:
            self.exit_reason = agent_runner_loop(
                client=self.client,
                system_prompt=build_system_prompt(),
                user_input=user_input,
                handler=self.handler,
                tools_schema=TOOLS_SCHEMA,
                max_turns=getattr(config, 'MAX_TURNS', 40),
                on_chunk=self._on_chunk,
            )
        except KeyboardInterrupt:
            self.exit_reason = {'result': 'INTERRUPTED'}
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.exit_reason = {'result': 'ERROR', 'msg': str(e)}
            self._on_chunk(f'\n**[Error]** {e}\n')
        finally:
            self.commit_response_so_far()
            try:
                archive_session(self.client, task_text, self.exit_reason)
            except Exception:
                pass
            self.save()
            self.turn_done_evt.set()

    def _on_chunk(self, chunk):
        if self.cancel_evt.is_set():
            raise KeyboardInterrupt('User cancelled')
        self.display_q.put({'chunk': chunk})
        with self._lock:
            self.response_chunks.append(chunk)

    def cancel(self):
        self.cancel_evt.set()
        try:
            self.ask_q.put_nowait('[Cancelled]')
        except queue.Full:
            pass

    def iter_text(self, text):
        self.run_or_answer(text)
        while True:
            try:
                item = self.display_q.get(timeout=0.1)
            except queue.Empty:
                if self.turn_done_evt.is_set():
                    break
                continue
            if 'chunk' in item:
                yield item['chunk']
            elif 'ask' in item:
                yield encode_ask_user_block(item['ask'])
            if self.turn_done_evt.is_set() and self.display_q.empty():
                break

    def is_running(self):
        return self.worker is not None and self.worker.is_alive()


class AgentService:
    def __init__(self):
        self.store = SessionStore(os.path.join(ROOT, 'memory', 'sessions'))
        self._sessions = {}
        self._lock = threading.RLock()

    def create_session(self, cwd=None):
        sid = str(uuid.uuid4())
        sess = AgentSession(self, sid, cwd=cwd)
        with self._lock:
            self._sessions[sid] = sess
        sess.save()
        return sess

    def get_session(self, sid=None, cwd=None):
        if not sid:
            return self.create_session(cwd=cwd)
        with self._lock:
            sess = self._sessions.get(sid)
            if sess:
                return sess
            sess = AgentSession(self, sid, cwd=cwd)
            loaded = self.store.load(sid)
            if loaded is not None:
                sess.restore_from(loaded)
            self._sessions[sid] = sess
            if loaded is None:
                sess.save()
            return sess

    def load_session(self, sid):
        with self._lock:
            sess = self._sessions.get(sid)
        loaded = self.store.load(sid)
        if loaded is None:
            return None
        if sess is None:
            sess = AgentSession(self, sid)
            sess.restore_from(loaded)
            with self._lock:
                self._sessions[sid] = sess
        return sess

    def list_sessions(self):
        rows = self.store.list_sessions()
        with self._lock:
            running = {sid: sess.is_running() for sid, sess in self._sessions.items()}
        for row in rows:
            row['running'] = running.get(row['session_id'], False)
        return rows

    def delete_session(self, sid):
        with self._lock:
            sess = self._sessions.pop(sid, None)
        if sess:
            sess.cancel()
        return self.store.delete(sid)

    def cancel_session(self, sid):
        with self._lock:
            sess = self._sessions.get(sid)
        if not sess:
            return False
        sess.cancel()
        return True

    def chat_text(self, sid, messages, cwd=None):
        sess = self.get_session(sid, cwd=cwd)
        text = last_user_text(messages)
        return sess, sess.iter_text(text)


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
