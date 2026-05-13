"""LLM 客户端 — 标准 OpenAI 兼容协议（默认 Qwen via DashScope）。

通过 openai 官方 SDK 调用任意 OpenAI 兼容端点：DashScope（Qwen）/ OpenAI /
DeepSeek / vLLM / Ollama / OpenRouter / Azure 等。

设计要点：
- chat() 是 generator：实时 yield 文本 chunk，最终 return Response
- history 走 OpenAI 原生消息格式（user/assistant/tool 三种 role）
- tools 走 {type:'function', function:{name, description, parameters}} 格式
- prompt cache 由服务端自动处理（Qwen/DeepSeek/OpenAI 都自动前缀缓存），客户端零配置
"""
import json
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolCallDelta:
    index: int
    id: str = ''
    name: str = ''
    name_delta: str = ''
    arguments_delta: str = ''
    arguments_text: str = ''


@dataclass
class Response:
    content: str = ''
    tool_calls: list = field(default_factory=list)
    stop_reason: str = ''
    usage: dict = field(default_factory=dict)


# ──────────────────────────── 历史裁剪 ──────────────────────────── #

def _estimate_tokens(messages):
    n = 0
    for m in messages:
        c = m.get('content') or ''
        if isinstance(c, str):
            n += len(c) // 4
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict):
                    n += len(str(blk.get('text', ''))) // 4
        for tc in m.get('tool_calls') or []:
            fn = tc.get('function', {}) if isinstance(tc, dict) else {}
            n += len(fn.get('arguments', '')) // 4
    return n


def trim_history(history, max_tokens):
    """超阈值则从最早消息开始删；保证不切断 assistant→tool 的配对。"""
    while _estimate_tokens(history) > max_tokens and len(history) > 2:
        # Drop messages until next one is a fresh user turn (avoid orphan tool messages)
        del history[0]
        while history and history[0].get('role') in ('tool', 'assistant'):
            del history[0]


# ──────────────────────────── 客户端主类 ──────────────────────────── #

class LLMClient:
    def __init__(self, api_key, api_base, model,
                 max_tokens=8192, history_trim_tokens=80000, timeout=300,
                 history_changed=None):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.max_tokens = max_tokens
        self.history_trim_tokens = history_trim_tokens
        self.timeout = timeout
        self.history = []
        self.history_changed = history_changed
        self._client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)

    def reset(self):
        self.history = []
        self._notify_history_changed()

    def _notify_history_changed(self):
        if not self.history_changed:
            return
        try:
            self.history_changed()
        except Exception:
            # Persistence callbacks are best-effort; the chat path should not
            # fail solely because a snapshot write failed.
            pass

    def chat(self, system, new_messages, tools):
        """yields text chunks during streaming, returns Response.

        new_messages: 本轮要追加的消息列表，每个元素是 OpenAI 标准消息：
          - 第一轮：[{'role':'user', 'content': str}]
          - 后续轮：[{'role':'tool','tool_call_id':..., 'content':...}, ...,
                     可选 {'role':'user','content': next_prompt}]
        """
        self.history.extend(new_messages)
        trim_history(self.history, self.history_trim_tokens)
        self._notify_history_changed()

        messages = [{'role': 'system', 'content': system}] + list(self.history)

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or None,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options={'include_usage': True},
            )
        except Exception as e:
            err = f'Request failed: {type(e).__name__}: {e}'
            yield f'\n[Error] {err}\n'
            return Response(content=err, stop_reason='error')

        content_parts = []
        tool_acc = {}                    # index → {id, name, args}
        finish_reason = None
        warn = None
        usage_dict = {}

        try:
            for chunk in stream:
                chunk_usage = getattr(chunk, 'usage', None)
                if chunk_usage is not None:
                    usage_dict = {
                        'prompt_tokens': getattr(chunk_usage, 'prompt_tokens', 0) or 0,
                        'completion_tokens': getattr(chunk_usage, 'completion_tokens', 0) or 0,
                        'total_tokens': getattr(chunk_usage, 'total_tokens', 0) or 0,
                    }
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, 'delta', None)
                if delta is not None:
                    txt = getattr(delta, 'content', None)
                    if txt:
                        content_parts.append(txt)
                        yield txt
                    tcs = getattr(delta, 'tool_calls', None)
                    if tcs:
                        for tc in tcs:
                            idx = getattr(tc, 'index', 0) or 0
                            slot = tool_acc.setdefault(idx, {'id': '', 'name': '', 'args': ''})
                            name_delta = ''
                            args_delta = ''
                            if getattr(tc, 'id', None):
                                slot['id'] = tc.id
                            fn = getattr(tc, 'function', None)
                            if fn is not None:
                                if getattr(fn, 'name', None):
                                    slot['name'] = fn.name
                                    name_delta = fn.name
                                if getattr(fn, 'arguments', None):
                                    args_delta = fn.arguments
                                    slot['args'] += args_delta
                            if getattr(tc, 'id', None) or name_delta or args_delta:
                                yield ToolCallDelta(
                                    index=idx,
                                    id=slot['id'],
                                    name=slot['name'],
                                    name_delta=name_delta,
                                    arguments_delta=args_delta,
                                    arguments_text=slot['args'],
                                )
                if getattr(choice, 'finish_reason', None):
                    finish_reason = choice.finish_reason
        except Exception as e:
            warn = f'\n[!! 流异常中断: {type(e).__name__}: {e} !!]'
            yield warn

        if finish_reason == 'length':
            warn = '\n[!! 响应被截断: max_tokens 上限 !!]'
            content_parts.append(warn)
            yield warn

        full_content = ''.join(content_parts)

        # 写回 assistant 消息到 history（含 tool_calls）
        ordered_idxs = sorted(tool_acc.keys())
        asst_msg = {'role': 'assistant', 'content': full_content or ''}
        if ordered_idxs:
            asst_msg['tool_calls'] = [{
                'id': tool_acc[i]['id'],
                'type': 'function',
                'function': {
                    'name': tool_acc[i]['name'],
                    'arguments': tool_acc[i]['args'] or '{}',
                },
            } for i in ordered_idxs]
        self.history.append(asst_msg)
        self._notify_history_changed()

        # 构造 Response
        resp = Response(content=full_content, stop_reason=finish_reason or '', usage=usage_dict)
        for i in ordered_idxs:
            slot = tool_acc[i]
            try:
                inp = json.loads(slot['args']) if slot['args'] else {}
            except Exception:
                inp = {'_raw_args': slot['args']}
            resp.tool_calls.append(ToolCall(id=slot['id'], name=slot['name'], input=inp))
        return resp
