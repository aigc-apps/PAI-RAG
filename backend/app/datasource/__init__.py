"""Knowledge-base data sources.

Adapters (``discover``/``fetch``/``emit``) are pure — config in, documents out,
no DB. Ported from the legacy pairag ``rag/datasource`` package and trimmed to the
backend's synchronous ingestion model (no Celery / file_store / manifest table).
"""
