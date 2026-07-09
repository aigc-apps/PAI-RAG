# Database migrations (Alembic)

Alembic is the single source of truth for the persistent database schema. It
works for both **SQLite** (local dev default) and **Postgres**, driven entirely
by the `DB_URL` env var — the same URL the app uses. Nothing is hard-coded in
`alembic.ini`; `alembic/env.py` reads the URL from the app settings.

Run all commands from `backend/`.

## Everyday flow

```bash
# 1. Change a model in app/models.py, then generate a migration:
alembic revision --autogenerate -m "add foo to bar"

# 2. REVIEW the generated file in alembic/versions/ before committing.
#    SQLite has loose type affinity, so autogenerate can emit spurious
#    type-change ops — delete anything that isn't a real change.

# 3. Apply it:
alembic upgrade head
```

`alembic downgrade -1` rolls back one revision.

## How it runs in the app

On boot, for a persistent DB, `app/db.py:migrate()` runs `alembic upgrade head`
automatically (a fresh DB gets the full baseline; an existing one gets new
revisions). Disable that and migrate from your deploy pipeline instead with:

```bash
AUTO_MIGRATE=false            # app skips boot migration
alembic upgrade head          # you run it explicitly
```

The in-memory store and the test suite do **not** use Alembic — they build the
schema directly with `SQLModel.metadata.create_all` (ephemeral, no history).

## Switching to Postgres

```bash
export DB_URL="postgresql+asyncpg://user:pass@host:5432/dbname"
alembic upgrade head          # same migrations, native ALTER (no batch mode)
```

SQLite migrations run in Alembic "batch mode" (rebuild-table) because SQLite
can't `ALTER` in place; Postgres alters natively. `env.py` picks the mode from
the dialect, so one set of migration scripts serves both.

## One-off target

`alembic -x db_url="sqlite+aiosqlite:///./scratch.db" upgrade head` overrides the
URL for a single command without touching the environment.
