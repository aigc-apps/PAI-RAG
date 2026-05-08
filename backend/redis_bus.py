"""Redis-backed run event stream, cancellation, and ask_user answer channel."""
import json
import os
import time
from typing import Iterator

import redis
import settings as config


DEFAULT_REDIS_URL = 'redis://127.0.0.1:6379/0'


def redis_url():
    return (
        os.environ.get('REDIS_URL')
        or os.environ.get('CELERY_BROKER_URL')
        or getattr(config, 'REDIS_URL', '')
        or getattr(config, 'CELERY_BROKER_URL', '')
        or DEFAULT_REDIS_URL
    )


class RedisBus:
    def __init__(self, url=None):
        self.url = url or redis_url()
        self.client = redis.Redis.from_url(self.url, decode_responses=True)
        self.event_ttl = int(getattr(config, 'RUN_EVENT_TTL_SECONDS', 24 * 60 * 60))
        self.ask_timeout = int(getattr(config, 'ASK_USER_TIMEOUT_SECONDS', 30 * 60))
        self.idle_timeout = int(getattr(config, 'RUN_IDLE_TIMEOUT_SECONDS', 60 * 60))

    @staticmethod
    def stream_key(run_id):
        return f'run:{run_id}:events'

    @staticmethod
    def cancel_key(run_id):
        return f'run:{run_id}:cancel'

    @staticmethod
    def answer_key(run_id):
        return f'run:{run_id}:answer'

    def publish_event(self, run_id, event):
        key = self.stream_key(run_id)
        event_id = self.client.xadd(key, {'event': json.dumps(event, ensure_ascii=False, default=str)})
        self.client.expire(key, self.event_ttl)
        return event_id

    def ping(self):
        return self.client.ping()

    def last_event_id(self, run_id):
        rows = self.client.xrevrange(self.stream_key(run_id), count=1)
        return rows[0][0] if rows else '0-0'

    def read_events(self, run_id, last_id='0-0', count=20, block_ms=1000):
        rows = self.client.xread({self.stream_key(run_id): last_id}, count=count, block=block_ms)
        events = []
        for _, messages in rows:
            for message_id, fields in messages:
                raw = fields.get('event') or '{}'
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    event = {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': raw}}
                events.append((message_id, event))
        return events

    def iter_events(self, run_id, last_id='0-0') -> Iterator[dict]:
        idle_started = time.time()
        while True:
            events = self.read_events(run_id, last_id=last_id)
            if not events:
                if time.time() - idle_started > self.idle_timeout:
                    yield {'sessionUpdate': 'done', 'stopReason': 'timeout'}
                    break
                continue
            idle_started = time.time()
            for message_id, event in events:
                last_id = message_id
                yield event
                if event.get('sessionUpdate') == 'done':
                    return

    def cancel(self, run_id):
        self.client.setex(self.cancel_key(run_id), self.event_ttl, '1')

    def is_cancelled(self, run_id):
        return self.client.exists(self.cancel_key(run_id)) > 0

    def push_answer(self, run_id, text):
        key = self.answer_key(run_id)
        self.client.rpush(key, text)
        self.client.expire(key, self.event_ttl)

    def wait_answer(self, run_id, timeout=None):
        item = self.client.blpop(self.answer_key(run_id), timeout=timeout or self.ask_timeout)
        if not item:
            raise TimeoutError('ask_user timed out waiting for answer')
        return item[1]
