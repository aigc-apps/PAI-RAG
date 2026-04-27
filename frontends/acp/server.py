"""Mini GenericAgent — Agent Client Protocol (ACP) frontend.

Run as: python -m frontends.acp.server
Speaks JSON-RPC 2.0 over stdio so editors like Zed can drive the agent.

Critical invariants:
- sys.__stdout__ is reserved for JSON-RPC; sys.stdout is redirected to a
  per-thread queue tee so all the agent's print() calls flow as session/update.
- One worker thread per session; survives across session/prompt RPCs so that
  ask_user can pause the loop and resume on the next user prompt.
- File ops are delegated to the client via fs/read_text_file & fs/write_text_file
  when the client advertises those capabilities; otherwise fall back to local IO.
"""
import os, sys, io, re, json, queue, uuid, threading, datetime, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from agent_loop import agent_runner_loop, StepOutcome
from tools import GenericHandler
from llm_client import LLMClient
from session_store import SessionStore
from skill_manager import (
    scan_skills, get_skills_prompt, get_use_skill_schema,
    match_skill, build_skill_user_input,
)
from frontends.acp.jsonrpc import JsonRpcServer

try:
    import config
except ImportError:
    sys.__stderr__.write('config.py not found. Run `cp config_template.py config.py` first.\n')
    sys.exit(1)


TOOLS_SCHEMA = json.load(open(os.path.join(ROOT, 'tools_schema.json'), encoding='utf-8'))
SYS_PROMPT_BASE = open(os.path.join(ROOT, 'prompts', 'sys_prompt.txt'), encoding='utf-8').read()
SKILLS = scan_skills(os.path.join(ROOT, 'skills'))
if SKILLS:
    TOOLS_SCHEMA.append(get_use_skill_schema())


# ── stdout redirection: per-thread tee to current session's queue ──

_thread_local = threading.local()


class _StdoutTee(io.TextIOBase):
    """Routes write()s to the current thread's bound display queue.
    Threads with no binding (e.g. the JSON-RPC read loop) get their output
    dropped to the real stderr for operator debugging.
    """
    def writable(self):
        return True

    def write(self, s):
        if not s:
            return 0
        q = getattr(_thread_local, 'display_q', None)
        if q is not None:
            q.put({'chunk': s})
        else:
            try:
                sys.__stderr__.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        pass


def build_system_prompt():
    idx_path = os.path.join(ROOT, 'memory', 'global_index.txt')
    idx = open(idx_path, encoding='utf-8').read() if os.path.exists(idx_path) else '(empty)'
    return SYS_PROMPT_BASE + '\n' + idx + get_skills_prompt(SKILLS)


def archive_session(client, task, exit_reason):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = os.path.join(ROOT, 'memory', 'L4_raw_sessions')
    os.makedirs(archive_dir, exist_ok=True)
    path = os.path.join(archive_dir, f'{ts}.md')
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'# Task ({ts})\n{task}\n\n')
            f.write(f'## Exit\n```json\n{json.dumps(exit_reason, ensure_ascii=False, default=str, indent=2)}\n```\n\n')
            f.write('## History\n```json\n')
            json.dump(client.history, f, ensure_ascii=False, default=str, indent=2)
            f.write('\n```\n')
    except Exception:
        pass


# ── Reverse RPC bridge: ask the client for fs operations ──

class AcpFsBridge:
    def __init__(self, rpc, session_id):
        self._rpc = rpc
        self._sid = session_id

    def read_text_file(self, path, line=None, limit=None):
        params = {'sessionId': self._sid, 'path': path}
        if line is not None:
            params['line'] = line
        if limit is not None:
            params['limit'] = limit
        result = self._rpc.send_request('fs/read_text_file', params)
        return (result or {}).get('content', '')

    def write_text_file(self, path, content):
        self._rpc.send_request('fs/write_text_file',
                               {'sessionId': self._sid, 'path': path, 'content': content})


# ── Agent handler with ACP-specific overrides ──

class AcpHandler(GenericHandler):
    def __init__(self, cwd, mini_agent_root, display_q, ask_q, fs, cancel_evt, turn_done_evt):
        super().__init__(cwd, mini_agent_root)
        self._dq = display_q
        self._aq = ask_q
        self._fs = fs                       # AcpFsBridge or None (fallback to local)
        self.cancel_evt = cancel_evt        # consumed by tools.code_run
        self._turn_done_evt = turn_done_evt

    # ── ask_user: emit question, signal turn pause, block on next prompt ──
    def do_ask_user(self, args, response):
        question = args.get('question', '请提供输入：')
        candidates = args.get('candidates') or []
        self._dq.put({'ask': {'question': question, 'candidates': candidates}})
        self._turn_done_evt.set()           # tell prompt RPC to return now
        answer = self._aq.get()             # block until next session/prompt
        if self.cancel_evt.is_set():
            raise KeyboardInterrupt('Cancelled while awaiting user input')
        if answer.strip().isdigit() and candidates and 1 <= int(answer) <= len(candidates):
            answer = candidates[int(answer) - 1]
        return StepOutcome({'status': 'answered', 'answer': answer},
                           next_prompt=f'用户回答：{answer}\n根据答案继续推进任务。')

    # ── file ops: delegate to client via reverse RPC if available ──
    def do_file_read(self, args, response):
        if self._fs is None:
            return super().do_file_read(args, response)
        path = self._abs(args.get('path', ''))
        try:
            text = self._fs.read_text_file(path,
                                           line=args.get('start'),
                                           limit=args.get('count'))
        except Exception as e:
            print(f'[Warn] fs/read_text_file failed ({e}); falling back to local IO')
            return super().do_file_read(args, response)
        next_prompt = self._anchor_prompt(skip=args.get('_index', 0) > 0)
        return StepOutcome(text, next_prompt=next_prompt)

    def do_file_write(self, args, response):
        if self._fs is None:
            return super().do_file_write(args, response)
        path = self._abs(args.get('path', ''))
        # Reuse parent's content extraction (file_content tag / fenced code)
        text = response.content or ''
        m = re.search(r'<file_content[^>]*>(.*)</file_content>', text, re.DOTALL)
        if m:
            blocks = m.group(1).strip()
        else:
            s, e = text.find('```'), text.rfind('```')
            blocks = text[text.find('\n', s) + 1:e].strip() if -1 < s < e else None
        if not blocks:
            return StepOutcome({'status': 'error',
                                'msg': '请把内容放进 <file_content>...</file_content> 或代码块'},
                               next_prompt='\n')
        mode = args.get('mode', 'overwrite')
        try:
            if mode in ('append', 'prepend'):
                # Read-modify-write: need both fs caps
                old = ''
                try:
                    old = self._fs.read_text_file(path)
                except Exception:
                    pass
                content = (blocks + old) if mode == 'prepend' else (old + blocks)
            else:
                content = blocks
            self._fs.write_text_file(path, content)
            print(f'[Status] ✅ {mode} via client fs ({len(content)} bytes)')
            return StepOutcome({'status': 'success', 'writed_bytes': len(content)},
                               next_prompt=self._anchor_prompt(skip=args.get('_index', 0) > 0))
        except Exception as ex:
            return StepOutcome({'status': 'error', 'msg': str(ex)}, next_prompt='\n')

    def do_file_patch(self, args, response):
        if self._fs is None:
            return super().do_file_patch(args, response)
        path = self._abs(args.get('path', ''))
        old = args.get('old_content', '')
        new = args.get('new_content', '')
        if not old:
            return StepOutcome({'status': 'error', 'msg': 'old_content 为空'},
                               next_prompt='\n')
        try:
            current = self._fs.read_text_file(path)
        except Exception as e:
            print(f'[Warn] fs/read_text_file failed ({e}); falling back to local IO')
            return super().do_file_patch(args, response)
        cnt = current.count(old)
        if cnt == 0:
            return StepOutcome({'status': 'error',
                                'msg': '未找到匹配。请先 file_read 确认当前内容，再分小段 patch。'},
                               next_prompt='\n')
        if cnt > 1:
            return StepOutcome({'status': 'error',
                                'msg': f'找到 {cnt} 处匹配，不唯一。请提供更长 old_content。'},
                               next_prompt='\n')
        try:
            self._fs.write_text_file(path, current.replace(old, new))
            print(f'[Status] ✅ patched via client fs')
            return StepOutcome({'status': 'success', 'msg': '文件局部修改成功'},
                               next_prompt=self._anchor_prompt(skip=args.get('_index', 0) > 0))
        except Exception as ex:
            return StepOutcome({'status': 'error', 'msg': str(ex)}, next_prompt='\n')


# ── Single session state ──

class AcpSession:
    def __init__(self, sid, cwd, rpc, has_fs, store):
        self.sid = sid
        self.cwd = cwd
        self.rpc = rpc
        self._store = store
        self.client = self._create_client()
        self.client.history_changed = self.save
        self.fs = AcpFsBridge(rpc, sid) if has_fs else None
        self.handler = None                  # GenericHandler — persists across prompts
        self.ui_msgs = []
        self._lock = threading.RLock()
        self.display_q = queue.Queue()
        self.ask_q = queue.Queue()
        self.cancel_evt = threading.Event()
        self.turn_done_evt = threading.Event()
        self.worker = None
        self.exit_reason = None
        self.pump_started = False
        self.response_chunks = []

    @staticmethod
    def _create_client():
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
            h = AcpHandler(self.cwd, ROOT, self.display_q, self.ask_q,
                           self.fs, self.cancel_evt, self.turn_done_evt)
            h.history_info = state.get('history_info', [])
            h.working = state.get('working', {})
            self.handler = h

    def save(self):
        try:
            with self._lock:
                llm_history = list(self.client.history)
                ui_msgs = list(self.ui_msgs)
            self._store.save(
                session_id=self.sid,
                llm_history=llm_history,
                ui_messages=ui_msgs,
                handler_state=self.snapshot_handler_state(),
            )
        except Exception:
            traceback.print_exc(file=sys.__stderr__)

    def append_ui_message(self, role, content):
        with self._lock:
            if (
                role == 'assistant'
                and self.ui_msgs
                and self.ui_msgs[-1].get('role') == 'assistant'
                and content.startswith(self.ui_msgs[-1].get('content') or '')
            ):
                self.ui_msgs[-1] = {'role': role, 'content': content}
            elif not (
                self.ui_msgs
                and self.ui_msgs[-1].get('role') == role
                and self.ui_msgs[-1].get('content') == content
            ):
                self.ui_msgs.append({'role': role, 'content': content})
        self.save()

    def start_pump(self):
        if self.pump_started:
            return
        self.pump_started = True
        threading.Thread(target=self._pump_loop, daemon=True).start()

    def _pump_loop(self):
        while True:
            item = self.display_q.get()
            try:
                self._emit_update(item)
            except Exception:
                # Swallow emit errors to keep pump alive
                pass

    def _emit_update(self, item):
        if 'chunk' in item:
            text = item['chunk']
            if not text:
                return
            self.rpc.send_notification('session/update', {
                'sessionId': self.sid,
                'update': {
                    'sessionUpdate': 'agent_message_chunk',
                    'content': {'type': 'text', 'text': text},
                },
            })
        elif 'ask' in item:
            ask = item['ask']
            text = f"\n❓ {ask['question']}\n"
            if ask.get('candidates'):
                for i, c in enumerate(ask['candidates'], 1):
                    text += f"  {i}. {c}\n"
            self.rpc.send_notification('session/update', {
                'sessionId': self.sid,
                'update': {
                    'sessionUpdate': 'agent_message_chunk',
                    'content': {'type': 'text', 'text': text},
                },
            })

    def run_or_answer(self, text):
        self.start_pump()
        if self.worker is not None and self.worker.is_alive():
            # Agent is paused on ask_user — feed the answer
            self.append_ui_message('user', text)
            self.turn_done_evt.clear()
            self.ask_q.put(text)
        else:
            self._spawn_worker(text)
        self.turn_done_evt.wait()
        return self._stop_reason()

    def cancel(self):
        self.cancel_evt.set()
        try:
            self.ask_q.put_nowait('[Cancelled]')
        except queue.Full:
            pass

    def _spawn_worker(self, text):
        # Skill matching follows the same path as the CLI entrypoint.
        sk, sk_args = match_skill(text, SKILLS)
        task_text = build_skill_user_input(sk, sk_args) if sk else text
        prev = self.handler
        new_h = AcpHandler(self.cwd, ROOT, self.display_q, self.ask_q,
                           self.fs, self.cancel_evt, self.turn_done_evt)
        if sk:
            new_h.working['active_skill'] = sk.name
            new_h.working['related_sop'] = f'skills/{sk.name}/SKILL.md'
        if prev is not None:
            new_h.history_info = list(prev.history_info)
            if 'key_info' in prev.working:
                ki = re.sub(r'\n\[SYSTEM\] 此为.*?工作记忆[。\n]*', '', prev.working['key_info'])
                new_h.working['key_info'] = ki
                ps = prev.working.get('passed_sessions', 0) + 1
                new_h.working['passed_sessions'] = ps
                new_h.working['key_info'] += (
                    f'\n[SYSTEM] 此为 {ps} 个对话前设置的key_info，'
                    f'若已在新任务，先更新或清除工作记忆。\n'
                )
        new_h.history_info.append(f"[USER]: {task_text[:200]}")
        self.handler = new_h
        user_input = task_text
        if prev is not None:
            user_input = new_h._anchor_prompt() + f'\n\n### 用户当前消息\n{task_text}'
        self.append_ui_message('user', text)
        self.cancel_evt.clear()
        self.turn_done_evt.clear()
        self.exit_reason = None
        self.response_chunks = []
        self.worker = threading.Thread(target=self._run_loop,
                                       args=(user_input, task_text), daemon=True)
        self.worker.start()

    def _run_loop(self, user_input, task_text):
        _thread_local.display_q = self.display_q   # bind tee for this thread
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
            traceback.print_exc(file=sys.__stderr__)
            self.exit_reason = {'result': 'ERROR', 'msg': str(e)}
            self.display_q.put({'chunk': f'\n**[Error]** {e}\n'})
            self.response_chunks.append(f'\n**[Error]** {e}\n')
        finally:
            full_response = ''.join(self.response_chunks)
            if full_response:
                self.append_ui_message('assistant', full_response)
            else:
                self.save()
            archive_session(self.client, task_text, self.exit_reason)
            self.turn_done_evt.set()

    def _on_chunk(self, chunk):
        if self.cancel_evt.is_set():
            raise KeyboardInterrupt('User cancelled')
        self.display_q.put({'chunk': chunk})
        self.response_chunks.append(chunk)

    def _stop_reason(self):
        if self.cancel_evt.is_set():
            return 'cancelled'
        if self.worker is not None and self.worker.is_alive():
            # Paused on ask_user — agent considers turn complete from ACP's POV
            return 'end_turn'
        result = (self.exit_reason or {}).get('result')
        if result == 'MAX_TURNS_EXCEEDED':
            return 'max_turn_requests'
        if result == 'INTERRUPTED':
            return 'cancelled'
        if result == 'ERROR':
            return 'refusal'
        return 'end_turn'

    def snapshot_handler_state(self):
        if self.handler is None:
            return None
        return {
            'history_info': list(self.handler.history_info),
            'working': dict(self.handler.working),
        }


# ── Top-level RPC dispatcher ──

class AcpServer:
    def __init__(self):
        self._sessions = {}
        self._client_caps = {}
        self._store = SessionStore(os.path.join(ROOT, 'memory', 'sessions'))
        self.rpc = JsonRpcServer(
            handlers={
                'initialize':       self.handle_initialize,
                'session/new':      self.handle_session_new,
                'session/load':     self.handle_session_load,
                'session/prompt':   self.handle_session_prompt,
                'session/cancel':   self.handle_session_cancel,
            },
            stream=sys.__stdout__,
            # session/prompt blocks waiting on worker; offload to thread so
            # the read loop keeps draining stdin (notably fs RPC responses).
            async_methods={'session/prompt'},
        )

    def handle_initialize(self, params):
        self._client_caps = params.get('clientCapabilities', {}) or {}
        return {
            'protocolVersion': 1,
            'agentCapabilities': {
                'loadSession': True,
                'promptCapabilities': {
                    'image': False,
                    'audio': False,
                    'embeddedContext': True,
                },
            },
            'authMethods': [],
        }

    def _has_fs(self):
        fs = (self._client_caps.get('fs') or {})
        return bool(fs.get('readTextFile')) and bool(fs.get('writeTextFile'))

    def handle_session_new(self, params):
        sid = str(uuid.uuid4())
        cwd = params.get('cwd') or os.getcwd()
        sess = AcpSession(sid, cwd, self.rpc, self._has_fs(), self._store)
        self._sessions[sid] = sess
        sess.save()
        return {'sessionId': sid}

    def handle_session_load(self, params):
        sid = params.get('sessionId')
        cwd = params.get('cwd') or os.getcwd()
        if not sid:
            raise ValueError('sessionId required')
        sess = AcpSession(sid, cwd, self.rpc, self._has_fs(), self._store)
        loaded = self._store.load(sid)
        if loaded is None:
            raise ValueError(f'Session not found: {sid}')
        sess.restore_from(loaded)
        self._sessions[sid] = sess
        # NOTE: v1 does not replay history as session/update notifications.
        # Zed will not show prior turns visually, but the agent has full context.
        return {}

    def handle_session_prompt(self, params):
        sid = params.get('sessionId')
        sess = self._sessions.get(sid)
        if sess is None:
            raise ValueError(f'Unknown session: {sid}')
        text = self._flatten_prompt(params.get('prompt'))
        stop_reason = sess.run_or_answer(text)
        # Persist after each turn (best-effort)
        sess.save()
        return {'stopReason': stop_reason}

    def handle_session_cancel(self, params):
        sid = params.get('sessionId')
        sess = self._sessions.get(sid)
        if sess is not None:
            sess.cancel()
        return {}

    @staticmethod
    def _flatten_prompt(content_blocks):
        parts = []
        for block in content_blocks or []:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            t = block.get('type')
            if t == 'text':
                parts.append(block.get('text', ''))
            elif t == 'resource_link':
                parts.append(f'[file: {block.get("uri", "")}]')
            elif t == 'resource':
                res = block.get('resource', {}) or {}
                parts.append(f'[file: {res.get("uri", "")}]\n{res.get("text", "")}')
        return '\n'.join(p for p in parts if p)

    def serve(self):
        self.rpc.serve_stdio()


def main():
    # Hijack stdout BEFORE serving so all of tools.py's print() flow into our tee.
    # sys.__stdout__ stays untouched and is exclusively used by JsonRpcServer.
    sys.stdout = _StdoutTee()
    try:
        AcpServer().serve()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
