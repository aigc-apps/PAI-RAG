"""LLM 客户端 — 标准 OpenAI 兼容协议（默认 Qwen via DashScope）。

通过 openai 官方 SDK 调用任意 OpenAI 兼容端点：DashScope（Qwen）/ OpenAI /
DeepSeek / vLLM / Ollama / OpenRouter / Azure 等。

设计要点：
- chat() 是 generator：实时 yield 文本 chunk，最终 return Response
- history 走 OpenAI 原生消息格式（user/assistant/tool 三种 role）
- tools 走 {type:'function', function:{name, description, parameters}} 格式
- prompt cache 由服务端自动处理（Qwen/DeepSeek/OpenAI 都自动前缀缓存），客户端零配置
- history 压缩走 ``context_compressor.compress_history``：先按 token 预算从
  中间删，再 sanitize tool_calls 配对，避免 byte-level 暴力截断把孤儿 tool
  message 留下来导致下次 400
- 失败处理走 ``error_classifier.classify_api_error``：按错误类型决定 retry /
  rotate / compress 三个维度，不再只看 status code 把 401/403/429 绑成一回事；
  退避用 ``retry_utils.jittered_backoff``，避免并发请求同步重试
"""
import json
import time
from dataclasses import dataclass, field

from openai import OpenAI

import provider_pool
from context_compressor import compress_history
from error_classifier import classify_api_error
from retry_utils import jittered_backoff


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


# ──────────────────────────── 客户端主类 ──────────────────────────── #

# 单次 chat() 的总尝试次数。1 次正常 + 最多 3 次重试。BAD_REQUEST /
# AUTH_PERMANENT 类错误 classified.retryable=False 会立即退出，不会浪费
# 这个上限。
_MAX_ATTEMPTS = 4


class LLMClient:
    def __init__(self, api_key, api_base, model,
                 max_tokens=8192, history_trim_tokens=80000, timeout=300,
                 history_changed=None, provider_name=None, key_id=None):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.max_tokens = max_tokens
        self.history_trim_tokens = history_trim_tokens
        self.timeout = timeout
        self.history = []
        self.history_changed = history_changed
        # Pool wiring is opt-in. When None, the key-rotation / report_failure
        # path is fully skipped and behaviour matches pre-pool clients.
        self.provider_name = provider_name
        self.key_id = key_id
        self._client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)

    def _rotate_key(self):
        """Swap to the next live key from the pool. Returns True on success.

        Called when the classifier flags ``should_rotate_credential`` and we
        still have attempts left. Pool ``acquire`` may itself raise (every
        key dead) — that's a hard fail and we let the caller fall through
        to error reporting.
        """
        if not self.provider_name:
            return False
        try:
            bundle = provider_pool.acquire(self.provider_name)
        except Exception:
            return False
        self.api_key = bundle.api_key
        self.api_base = bundle.api_base
        self.key_id = bundle.key_id
        self._client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
        return True

    def _feedback_failure(self, status):
        """Tell the pool what just happened. Best-effort — never let pool
        bookkeeping bring down the chat path."""
        if not self.provider_name or status is None:
            return
        try:
            provider_pool.report_failure(self.provider_name, self.key_id, status)
        except Exception:
            pass

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
        self.history = compress_history(self.history, max_tokens=self.history_trim_tokens)
        self._notify_history_changed()

        messages = [{'role': 'system', 'content': system}] + list(self.history)

        # Each attempt:
        #   1. fire the request
        #   2. on exception, classify it
        #   3. let the classifier tell us whether to retry, rotate the key,
        #      and/or shrink history; sleep with jittered backoff before the
        #      next try
        # BAD_REQUEST / AUTH_PERMANENT short-circuit (retryable=False);
        # CONTEXT_OVERFLOW shrinks ``messages`` for THIS retry only — we
        # don't permanently mutate ``self.history`` because the next chat()
        # call will rebuild it.
        stream = None
        last_exc = None
        last_classified = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                stream = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools or None,
                    max_tokens=self.max_tokens,
                    stream=True,
                    stream_options={'include_usage': True},
                )
                break
            except Exception as e:
                last_exc = e
                last_classified = classify_api_error(e)
                if last_classified.should_rotate_credential:
                    self._feedback_failure(last_classified.status_code)
                if not last_classified.retryable or attempt == _MAX_ATTEMPTS - 1:
                    break
                if last_classified.should_rotate_credential and not self._rotate_key():
                    break
                if last_classified.should_compress:
                    # Tighter budget for this retry only — give the model
                    # half the room and see if it goes through.
                    tighter = compress_history(
                        self.history,
                        max_tokens=max(self.history_trim_tokens // 2, 1),
                    )
                    messages = [{'role': 'system', 'content': system}] + list(tighter)
                time.sleep(jittered_backoff(attempt + 1))
        if stream is None:
            reason_label = last_classified.reason.value if last_classified else 'unknown'
            err = (f'Request failed after {_MAX_ATTEMPTS} attempts ({reason_label}): '
                   f'{type(last_exc).__name__}: {last_exc}')
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
            # Retrying mid-stream would corrupt partial output already yielded
            # to the caller, so just feed the status to the pool and let the
            # next chat() turn pick a fresh key if the current one is bad.
            mid_classified = classify_api_error(e)
            if mid_classified.should_rotate_credential:
                self._feedback_failure(mid_classified.status_code)
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


def make_llm_client(model_override=None, history_changed=None):
    """One-stop factory for an LLMClient that's wired into the provider pool.

    All four production construction sites (worker, agent_service, ACP server,
    background memory review) used to inline the same six-arg ``LLMClient(...)``
    call reading from ``settings`` + ``runtime_config``. Centralising here means
    adding a new credential rule (e.g. per-region routing) is a one-place change
    and the four call sites stay tiny.

    ``model_override`` wins over ``runtime_config.get_active_model()`` for the
    one chat session this client serves; the global active model is untouched.

    On total pool exhaustion (every key for the resolved provider is dead) we
    still hand back a client backed by the env-configured key — degraded mode
    is more useful than a 500 on every request, and the client's own retry
    loop will surface a clean error if even that key fails.
    """
    # Imports inside the function avoid the runtime_config → settings →
    # llm_client import cycle on module load.
    import runtime_config
    import settings as config

    model = model_override or runtime_config.get_active_model()
    provider = provider_pool.resolve_provider(model)
    bundle = None
    try:
        bundle = provider_pool.acquire(provider)
    except (provider_pool.NoLiveKeyError, provider_pool.UnknownProviderError):
        bundle = None

    if bundle is not None:
        api_key = bundle.api_key
        api_base = bundle.api_base
        provider_name = bundle.provider
        key_id = bundle.key_id
    else:
        api_key = getattr(config, 'API_KEY', '') or ''
        api_base = getattr(config, 'API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        provider_name = None
        key_id = None

    return LLMClient(
        api_key=api_key,
        api_base=api_base,
        model=model,
        max_tokens=getattr(config, 'MAX_TOKENS', 8192),
        history_trim_tokens=getattr(config, 'HISTORY_TRIM_TOKENS', 80000),
        timeout=getattr(config, 'TIMEOUT', 300),
        history_changed=history_changed,
        provider_name=provider_name,
        key_id=key_id,
    )
