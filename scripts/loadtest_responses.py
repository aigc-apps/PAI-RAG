"""Tiny asyncio load tester for /v1/responses.

Usage:  python scripts/loadtest_responses.py --concurrency 8 --total 32
"""
import argparse, asyncio, json, statistics, time
import httpx

URL = 'http://localhost:9000/v1/responses'
_MULTITOOL_INPUT = (
    'Execute these steps EXACTLY, one tool call per step, do not batch. '
    'Step 1: call update_todo with items=[{"id":"1","content":"alpha","status":"pending"}]. '
    'Step 2: call update_working_checkpoint with key_info="started". '
    'Step 3: call update_todo with items=[{"id":"1","content":"alpha","status":"in_progress"},{"id":"2","content":"beta","status":"pending"}]. '
    'Step 4: call update_working_checkpoint with key_info="midway". '
    'Step 5: call update_todo with items=[{"id":"1","content":"alpha","status":"completed"},{"id":"2","content":"beta","status":"completed"}]. '
    'Step 6: call final_report with report_markdown="done" and summary="finished six steps".'
)
PAYLOAD = {
    'model': 'qwen-plus',
    'input': _MULTITOOL_INPUT,
    'stream': False,
    'store': True,
}


async def one(client, idx):
    t0 = time.perf_counter()
    try:
        r = await client.post(URL, json=PAYLOAD, timeout=120.0)
        dt = time.perf_counter() - t0
        ok = r.status_code == 200
        return idx, ok, r.status_code, dt, (r.text[:120] if not ok else '')
    except Exception as e:
        return idx, False, -1, time.perf_counter() - t0, repr(e)[:120]


async def main(concurrency, total):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def guarded(client, i):
        async with sem:
            return await one(client, i)

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        wall0 = time.perf_counter()
        results = await asyncio.gather(*[guarded(client, i) for i in range(total)])
        wall = time.perf_counter() - wall0

    lats = [r[3] for r in results if r[1]]
    errs = [r for r in results if not r[1]]
    lats.sort()

    def pct(p):
        if not lats:
            return float('nan')
        k = max(0, min(len(lats) - 1, int(round(p / 100 * (len(lats) - 1)))))
        return lats[k]

    print(json.dumps({
        'concurrency': concurrency,
        'total': total,
        'wall_s': round(wall, 2),
        'ok': len(lats),
        'err': len(errs),
        'qps': round(len(lats) / wall, 2) if wall else 0,
        'p50': round(pct(50), 2) if lats else None,
        'p95': round(pct(95), 2) if lats else None,
        'p99': round(pct(99), 2) if lats else None,
        'mean': round(statistics.mean(lats), 2) if lats else None,
        'min': round(min(lats), 2) if lats else None,
        'max': round(max(lats), 2) if lats else None,
    }, indent=2))

    if errs:
        print('--- first 3 errors ---')
        for e in errs[:3]:
            print(e)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--concurrency', '-c', type=int, default=4)
    ap.add_argument('--total', '-n', type=int, default=16)
    args = ap.parse_args()
    asyncio.run(main(args.concurrency, args.total))
