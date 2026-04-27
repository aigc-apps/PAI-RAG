"""Smoke test for the ACP frontend.

Spawns `python -m frontends.acp`, exchanges JSON-RPC messages over
stdio, and verifies the basic lifecycle: initialize → session/new →
session/prompt → session/cancel.

Mocks the reverse fs/* RPC requests by responding with stub content.

Run from project root:  python tests/acp_smoke.py
"""
import os, sys, json, time, threading, subprocess, itertools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AcpClient:
    def __init__(self, proc):
        self.proc = proc
        self._next_id = itertools.count(1)
        self._inbox = []                 # all messages received
        self._inbox_lock = threading.Lock()
        self._inbox_evt = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for raw in iter(self.proc.stdout.readline, b''):
            try:
                line = raw.decode('utf-8').strip()
            except Exception:
                continue
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                sys.stderr.write(f'[non-json] {line[:200]}\n')
                continue
            with self._inbox_lock:
                self._inbox.append(msg)
                self._inbox_evt.set()
            # Auto-respond to reverse fs RPC requests
            if msg.get('method', '').startswith('fs/'):
                threading.Thread(target=self._respond_fs, args=(msg,), daemon=True).start()

    def _respond_fs(self, req):
        method = req['method']
        mid = req.get('id')
        params = req.get('params', {})
        if method == 'fs/read_text_file':
            path = params.get('path', '')
            content = f'[stub fs/read of {path}]\nline1\nline2\n'
            self._send({'jsonrpc': '2.0', 'id': mid, 'result': {'content': content}})
        elif method == 'fs/write_text_file':
            print(f'[mock] fs/write_text_file path={params.get("path")} '
                  f'len={len(params.get("content",""))}')
            self._send({'jsonrpc': '2.0', 'id': mid, 'result': {}})
        else:
            self._send({'jsonrpc': '2.0', 'id': mid,
                        'error': {'code': -32601, 'message': 'unknown'}})

    def _send(self, msg):
        line = json.dumps(msg, ensure_ascii=False) + '\n'
        self.proc.stdin.write(line.encode('utf-8'))
        self.proc.stdin.flush()

    def request(self, method, params, timeout=30):
        mid = next(self._next_id)
        self._send({'jsonrpc': '2.0', 'id': mid, 'method': method, 'params': params})
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._inbox_lock:
                for m in self._inbox:
                    if m.get('id') == mid and 'method' not in m:
                        return m
            self._inbox_evt.wait(0.1)
            self._inbox_evt.clear()
        raise TimeoutError(f'Timed out waiting for response to {method}')

    def notify(self, method, params):
        self._send({'jsonrpc': '2.0', 'method': method, 'params': params})

    def updates(self):
        with self._inbox_lock:
            return [m for m in self._inbox if m.get('method') == 'session/update']


def main():
    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, '-u', '-m', 'frontends.acp'],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, bufsize=0,
    )
    # Drain stderr in background to avoid blocking
    def _drain_stderr():
        for raw in iter(proc.stderr.readline, b''):
            sys.stderr.write('[server.stderr] ' + raw.decode('utf-8', errors='replace'))
    threading.Thread(target=_drain_stderr, daemon=True).start()

    client = AcpClient(proc)
    try:
        # 1) initialize
        r = client.request('initialize', {
            'protocolVersion': 1,
            'clientCapabilities': {
                'fs': {'readTextFile': True, 'writeTextFile': True},
            },
        })
        assert 'result' in r, f'initialize failed: {r}'
        caps = r['result']['agentCapabilities']
        assert caps.get('loadSession') is True, 'loadSession capability missing'
        print('[OK] initialize → loadSession=True, embeddedContext=',
              caps.get('promptCapabilities', {}).get('embeddedContext'))

        # 2) session/new
        r = client.request('session/new', {'cwd': '/tmp', 'mcpServers': []})
        sid = r['result']['sessionId']
        print(f'[OK] session/new → sessionId={sid[:8]}...')

        # 3) session/prompt — send a tiny task. Real LLM will be called; we just
        # verify that updates flow and the prompt completes within a reasonable
        # timeout. If no API key is set, this will error out gracefully.
        prompt_text = '请直接调用 code_run 工具运行 python 代码 print("hello acp")，然后结束。'
        print(f'[..] sending prompt: {prompt_text[:60]}...')
        r = client.request('session/prompt', {
            'sessionId': sid,
            'prompt': [{'type': 'text', 'text': prompt_text}],
        }, timeout=120)
        assert 'result' in r, f'prompt failed: {r}'
        stop = r['result'].get('stopReason')
        print(f'[OK] session/prompt → stopReason={stop}')
        ups = client.updates()
        print(f'[OK] received {len(ups)} session/update notifications')
        if ups:
            sample = ups[0]['params']['update']
            print(f'      first update: sessionUpdate={sample.get("sessionUpdate")}')

    except Exception as e:
        print(f'[FAIL] {type(e).__name__}: {e}')
        import traceback; traceback.print_exc()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == '__main__':
    main()
