"""Minimal bidirectional JSON-RPC 2.0 over stdio (ndjson).

- Reads one JSON message per line from stdin
- Dispatches inbound requests/notifications to handler functions
- Routes inbound responses to in-flight outgoing requests (Event-based wait)
- Outbound writes go to sys.__stdout__ (real stdout); sys.stdout can be safely
  redirected by the host process for capturing arbitrary print() output.
"""
import sys, json, threading, itertools


class JsonRpcServer:
    def __init__(self, handlers, stream=None, async_methods=None):
        self._handlers = handlers
        self._pending = {}                # outgoing id -> slot dict
        self._pending_lock = threading.Lock()
        self._next_id = itertools.count(1)
        self._write_lock = threading.Lock()
        self._stream = stream or sys.__stdout__
        self._async = set(async_methods or [])

    def serve_stdio(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._dispatch(msg)

    def _dispatch(self, msg):
        if 'method' in msg:
            self._handle_inbound(msg)
        elif 'id' in msg:
            self._route_response(msg)

    def _handle_inbound(self, msg):
        mid = msg.get('id')
        method = msg['method']
        params = msg.get('params', {}) or {}
        handler = self._handlers.get(method)
        if handler is None:
            if mid is not None:
                self._send({'jsonrpc': '2.0', 'id': mid,
                            'error': {'code': -32601, 'message': f'Method not found: {method}'}})
            return
        if mid is None or method in self._async:
            threading.Thread(target=self._invoke, args=(handler, params, mid),
                             daemon=True).start()
        else:
            self._invoke(handler, params, mid)

    def _invoke(self, handler, params, mid):
        try:
            result = handler(params)
            if mid is not None:
                self._send({'jsonrpc': '2.0', 'id': mid,
                            'result': result if result is not None else {}})
        except Exception as e:
            if mid is not None:
                self._send({'jsonrpc': '2.0', 'id': mid,
                            'error': {'code': -32000, 'message': f'{type(e).__name__}: {e}'}})

    def _route_response(self, msg):
        mid = msg.get('id')
        with self._pending_lock:
            slot = self._pending.pop(mid, None)
        if slot is None:
            return
        slot['result'] = msg.get('result')
        slot['error'] = msg.get('error')
        slot['event'].set()

    def send_request(self, method, params, timeout=30):
        mid = next(self._next_id)
        slot = {'event': threading.Event(), 'result': None, 'error': None}
        with self._pending_lock:
            self._pending[mid] = slot
        self._send({'jsonrpc': '2.0', 'id': mid, 'method': method, 'params': params})
        if not slot['event'].wait(timeout):
            with self._pending_lock:
                self._pending.pop(mid, None)
            raise TimeoutError(f'JSON-RPC request "{method}" timed out after {timeout}s')
        if slot['error'] is not None:
            raise RuntimeError(f'RPC error from peer: {slot["error"]}')
        return slot['result']

    def send_notification(self, method, params):
        self._send({'jsonrpc': '2.0', 'method': method, 'params': params})

    def _send(self, msg):
        line = json.dumps(msg, ensure_ascii=False, default=str) + '\n'
        with self._write_lock:
            self._stream.write(line)
            self._stream.flush()
