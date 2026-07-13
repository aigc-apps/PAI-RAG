# Alembic Runtime Database URL Design

## Goal

Allow percent-encoded credentials in `DB_URL` during programmatic Alembic
migrations without routing the runtime URL through ConfigParser interpolation.

## Design

`backend/app/db.py::_alembic_config()` stores the runtime URL in
`Config.attributes["db_url"]`. Static Alembic settings such as
`script_location` remain regular main options.

`backend/alembic/env.py::_db_url()` resolves the URL in this order:

1. CLI `-x db_url=...` override.
2. Programmatic `Config.attributes["db_url"]` value.
3. Application `DB_URL` setting.

The old `sqlalchemy.url` injected main-option path is removed. Consequently,
runtime URLs containing `%40`, `%25`, or other valid percent escapes reach
SQLAlchemy unchanged and never enter ConfigParser interpolation.

## Compatibility and Security

Direct `alembic upgrade head` continues to read `DB_URL` through application
settings. The existing one-off `-x db_url=...` override remains highest
priority. No URL or credential is logged, decoded, or rewritten.

## Testing

Add a migration configuration regression test using a PostgreSQL URL whose
password contains percent-encoded `@` and `%` characters. Assert that creating
the Alembic config succeeds and that its runtime attribute exactly equals the
input URL. Retain the existing migration suite to cover SQLite execution.
