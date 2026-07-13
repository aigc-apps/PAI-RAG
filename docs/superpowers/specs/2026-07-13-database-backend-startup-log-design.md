# Database Backend Startup Log Design

## Goal

Log the active database backend during application startup so operators can
immediately distinguish PostgreSQL from SQLite without exposing connection
credentials.

## Design

Add a small helper in `backend/app/lean_main.py` that derives a display label
from `STORE_BACKEND` and `DB_URL`. The application lifespan calls it after
settings are loaded and before migrations begin, producing one Loguru message:

```text
[db] database backend = postgresql
```

When `STORE_BACKEND=memory`, the label is `sqlite (memory)` because the service
constructs an in-memory SQLite engine regardless of `DB_URL`. Persistent URLs
use SQLAlchemy URL parsing to obtain the backend name and exclude driver
suffixes such as `+asyncpg` and `+aiosqlite`.

## Security and Error Handling

The log never includes the URL, username, password, hostname, port, or database
name. Settings validation already supplies a non-empty `DB_URL`; if parsing
fails, startup follows the existing failure path rather than inventing a
potentially misleading backend label.

## Testing

Add focused unit tests for PostgreSQL and in-memory modes. Tests intercept the
Loguru message and assert the safe backend label is present while credentials
and host information are absent.
