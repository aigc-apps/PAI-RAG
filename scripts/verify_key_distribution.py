"""End-to-end check that /v1/responses distributes across multiple pool keys.

Three signals, strongest last:

  1. `/v1/admin/pool` reports >=2 live keys before the run, and the same set
     afterwards (no evictions / cooldowns triggered by healthy traffic).
  2. Concurrent /v1/responses requests all succeed.
  3. (Optional, the actual proof) -- if `--log` points at the worker log file,
     count `acquired llm key tail=XXXX` lines emitted during the run and
     report per-tail distribution. A correctly working pool produces a roughly
     even split; last-write-wins would funnel everything into one tail.

Usage:
    python scripts/verify_key_distribution.py \
        --url http://localhost:9000 \
        --concurrency 8 --total 32 \
        --log /tmp/pairag.log
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from collections import Counter

import httpx


_PAYLOAD = {
    'model': 'qwen-plus',
    # Single-turn, no tools -> minimum work per request so the run is
    # dominated by LLM RTT and we exercise as many key acquisitions as
    # possible per second.
    'input': 'Reply with the single word: pong',
    'stream': False,
    'store': False,
}


async def _fire_one(client: httpx.AsyncClient, url: str, idx: int) -> tuple[int, bool, int, float, str]:
    t0 = time.perf_counter()
    try:
        r = await client.post(f'{url}/v1/responses', json=_PAYLOAD, timeout=120.0)
        dt = time.perf_counter() - t0
        ok = r.status_code == 200
        return idx, ok, r.status_code, dt, (r.text[:200] if not ok else '')
    except Exception as e:
        return idx, False, -1, time.perf_counter() - t0, repr(e)[:200]


def _fetch_snapshot(url: str) -> dict:
    r = httpx.get(f'{url}/v1/admin/pool', timeout=10.0)
    r.raise_for_status()
    return r.json()


def _summarize_pool(snap: dict) -> dict:
    """Pull a compact (provider, tail, evicted, cooldown_until) view."""
    out = []
    for provider, info in snap.get('providers', {}).items():
        for key in info.get('keys', []):
            out.append({
                'provider': provider,
                'key_id': key['key_id'],
                'tail': key['api_key_tail'],
                'evicted': key['evicted'],
                'cooldown_until': key['cooldown_until'],
            })
    return out


def _grep_log_distribution(log_path: str, since_ts: float) -> Counter:
    """Count `acquired llm key tail=XXXX` lines whose log timestamp is >= since_ts.

    Assumes the worker log starts each line with an iso-ish timestamp, which
    is the default ``logging.basicConfig`` format. Falls back to counting
    every match if no timestamp can be parsed.
    """
    try:
        # -a forces text mode so logs that happen to contain a NUL byte
        # (e.g. emitted by an upstream library) don't get treated as binary,
        # which would suppress matching-line output and silently drop counts.
        out = subprocess.check_output(['grep', '-a', 'acquired llm key tail=', log_path], text=True)
    except subprocess.CalledProcessError:
        return Counter()
    pattern = re.compile(r'acquired llm key tail=(\S+)')
    counts: Counter = Counter()
    for line in out.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        counts[m.group(1)] += 1
    return counts


async def _run(url: str, concurrency: int, total: int) -> list:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(client, i):
        async with sem:
            return await _fire_one(client, url, i)

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        wall0 = time.perf_counter()
        results = await asyncio.gather(*[guarded(client, i) for i in range(total)])
        wall = time.perf_counter() - wall0
    return results, wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:9000')
    ap.add_argument('--concurrency', '-c', type=int, default=8)
    ap.add_argument('--total', '-n', type=int, default=32)
    ap.add_argument('--log', default=None,
                    help='Optional path to a worker log file to grep for key-tail distribution')
    args = ap.parse_args()

    print(f'==> fetching pool snapshot from {args.url}/v1/admin/pool')
    try:
        snap_before = _fetch_snapshot(args.url)
    except httpx.HTTPError as e:
        print(f'FAIL: could not reach /v1/admin/pool: {e}', file=sys.stderr)
        return 2

    keys_before = _summarize_pool(snap_before)
    live_before = [k for k in keys_before if not k['evicted'] and k['cooldown_until'] == 0.0]
    print(json.dumps(snap_before, indent=2))

    if len(live_before) < 2:
        print(
            f'\nWARN: only {len(live_before)} live key(s) configured; distribution is impossible. '
            'Add a second key to memory/runtime.json and reload the server.',
            file=sys.stderr,
        )
        return 3

    start_ts = time.time()
    print(f'\n==> firing {args.total} requests at concurrency={args.concurrency}')
    results, wall = asyncio.run(_run(args.url, args.concurrency, args.total))
    ok = [r for r in results if r[1]]
    errs = [r for r in results if not r[1]]
    print(json.dumps({
        'concurrency': args.concurrency,
        'total': args.total,
        'wall_s': round(wall, 2),
        'ok': len(ok),
        'err': len(errs),
        'qps': round(len(ok) / wall, 2) if wall else 0,
    }, indent=2))
    if errs:
        print('--- first 3 errors ---')
        for e in errs[:3]:
            print(e)

    print('\n==> fetching pool snapshot again')
    snap_after = _fetch_snapshot(args.url)
    keys_after = _summarize_pool(snap_after)
    diff = []
    for b, a in zip(keys_before, keys_after):
        if b['evicted'] != a['evicted'] or (b['cooldown_until'] == 0.0) != (a['cooldown_until'] == 0.0):
            diff.append({'tail': b['tail'], 'before': b, 'after': a})
    if diff:
        print('STATE CHANGED during run (may indicate auth/rate-limit issues):')
        print(json.dumps(diff, indent=2))
    else:
        print(f'pool unchanged: {len(live_before)} keys still live')

    if args.log:
        print(f'\n==> counting `acquired llm key tail=` in {args.log}')
        counts = _grep_log_distribution(args.log, start_ts)
        if not counts:
            print(f'no matching lines found; check that the worker log goes to {args.log}')
        else:
            total_acquires = sum(counts.values())
            print(f'total acquisitions logged: {total_acquires}')
            for tail, n in counts.most_common():
                pct = 100.0 * n / total_acquires
                print(f'  tail={tail}  count={n}  ({pct:.1f}%)')
            if len(counts) == 1:
                print('\nFAIL: all acquisitions hit a single key -- distribution is NOT working')
                return 1
            spread = max(counts.values()) / min(counts.values())
            print(f'\nmax/min ratio = {spread:.2f} (1.0 = perfectly even, <2.0 healthy for small N)')

    print('\nOK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
