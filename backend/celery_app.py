"""Celery application for distributed agent runs."""
import os

from celery import Celery
import settings as config


broker_url = (
    os.environ.get('CELERY_BROKER_URL')
    or os.environ.get('REDIS_URL')
    or getattr(config, 'CELERY_BROKER_URL', '')
    or getattr(config, 'REDIS_URL', '')
    or 'redis://127.0.0.1:6379/0'
)

celery_app = Celery('pai_rag_agent', broker=broker_url, backend=None, include=['backend.worker', 'backend.background_review'])
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_ignore_result=True,
    broker_connection_retry_on_startup=True,
)
