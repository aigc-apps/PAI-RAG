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


# ──────────────────────────── 工具参数修复 ──────────────────────────── #

def _repair_truncated_json(s):
    """把流式截断的 tool args JSON 尽力补齐成可解析对象。

    实测在 OpenAI-compatible 后端上,模型偶尔会在 finish_reason=tool_calls 的情况下
    少打一两个收尾括号(`}` 或 `]`),`json.loads` 直接报 `Expecting ',' delimiter`,
    整个 args dict 就被丢给 _raw_args 兜底了 —— 下游工具拿不到任何字段。

    覆盖三种最常见的截断:
      - 缺收尾括号        `{"a": [1,2,3]`        -> `{"a": [1,2,3]}`
      - 字符串内截断      `{"a": "hel`           -> `{"a":"hel"}`
      - 末尾悬空标点      `{"a":1,` 或 `{"a":`   -> `{"a":1}` / `{"a":null}`

    扫描时正确处理 `\\"` / `\\\\` / Unicode 转义,避免把字符串里的 `}` 误数为收尾。
    成功返回 dict/list,失败返回 None — 不抛异常。
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # 简单情形:已经合法。OpenAI tool args 必须是对象/数组,标量(数字/字符串/null)
    # 都视为非法 —— 上层 dispatch 会调 args.get(...) 直接崩。
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (dict, list)):
            return parsed
        return None
    except Exception:
        pass
    if s[0] not in '{[':
        return None

    stack = []                     # 元素是 '{' 或 '['
    in_string = False
    escape = False
    last_significant_idx = -1      # 最后一个非空白、非"开放结构性符号"字符的位置
    for idx, ch in enumerate(s):
        if escape:
            escape = False
            last_significant_idx = idx
            continue
        if in_string:
            if ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            last_significant_idx = idx
            continue
        if ch == '"':
            in_string = True
            last_significant_idx = idx
            continue
        if ch in '{[':
            stack.append(ch)
            continue
        if ch in '}]':
            if stack and ((ch == '}' and stack[-1] == '{') or (ch == ']' and stack[-1] == '[')):
                stack.pop()
            else:
                return None        # 结构错位,放弃
            last_significant_idx = idx
            continue
        if ch.isspace():
            continue
        last_significant_idx = idx

    repaired = s
    if in_string:
        # 1) 字符串没收尾,先补一个 `"`。如果末尾恰好是单独一个 `\\`,
        #    再补 `"` 会变成 `\\"` 转义,反而引发新错误 —— 直接砍掉那个反斜杠。
        if repaired.endswith('\\'):
            repaired = repaired[:-1]
        repaired += '"'
    else:
        # 2) 处理悬空标点。只看 last_significant_idx 之后的尾巴。
        tail_start = last_significant_idx + 1
        tail = repaired[tail_start:]
        head = repaired[:tail_start]
        # head 末尾可能是 `,` 或 `:` —— 这两种都意味着后面本应跟个值。
        head = head.rstrip()
        if head.endswith(','):
            head = head[:-1].rstrip()
        elif head.endswith(':'):
            head = head + ' null'
        repaired = head + tail.rstrip()

    # 3) 反向补齐 stack。
    closers = {'{': '}', '[': ']'}
    while stack:
        repaired += closers[stack.pop()]

    try:
        return json.loads(repaired)
    except Exception:
        return None


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
        for list_idx, i in enumerate(ordered_idxs):
            slot = tool_acc[i]
            raw = slot['args'] or ''
            try:
                inp = json.loads(raw) if raw else {}
            except Exception:
                # 模型偶发地少打收尾括号/单引号 —— 试着补齐再解一次。修复成功就把
                # 合法 JSON 写回 history,免得下一轮 LLM 重读自己半截的输出。
                repaired = _repair_truncated_json(raw)
                if repaired is not None:
                    inp = repaired
                    asst_msg['tool_calls'][list_idx]['function']['arguments'] = json.dumps(
                        repaired, ensure_ascii=False
                    )
                else:
                    inp = {'_raw_args': raw}
            resp.tool_calls.append(ToolCall(id=slot['id'], name=slot['name'], input=inp))
        return resp
