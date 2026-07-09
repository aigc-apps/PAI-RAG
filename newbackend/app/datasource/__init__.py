"""Knowledge-base data sources.

Adapters (``discover``/``fetch``/``emit``) are pure — config in, documents out,
no DB. Ported from the older ``backend/rag/datasource`` package and trimmed to the
newbackend's synchronous ingestion model (no Celery / file_store / manifest table).
"""
