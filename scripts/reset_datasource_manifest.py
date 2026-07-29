"""Reset a datasource manifest so the next sync re-ingests all documents.

Usage:
    cd backend
    python ../scripts/reset_datasource_manifest.py <datasource_id>

This deletes all DataSourceDocumentEntity rows for the given datasource.
After running, trigger a sync from the UI or API to re-pull everything.
"""

import asyncio
import os
import sys

import dotenv
dotenv.load_dotenv()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python reset_datasource_manifest.py <datasource_id>")
        sys.exit(1)

    datasource_id = sys.argv[1]

    from db.db_context import create_db_session
    from db.models.knowledgebase.datasource import DataSourceDocumentEntity
    from sqlmodel import select, delete
    from loguru import logger

    async with create_db_session() as session:
        # Count existing rows
        count_result = await session.exec(
            select(DataSourceDocumentEntity).where(
                DataSourceDocumentEntity.datasource_id == datasource_id
            )
        )
        rows = count_result.all()
        print(f"Found {len(rows)} manifest rows for datasource {datasource_id}")

        if not rows:
            print("Nothing to delete.")
            return

        # Show what will be deleted
        for r in rows:
            print(f"  - doc_id={r.doc_id}  title={r.title}  status={r.doc_status}")

        # Delete
        stmt = delete(DataSourceDocumentEntity).where(
            DataSourceDocumentEntity.datasource_id == datasource_id
        )
        result = await session.exec(stmt)
        await session.commit()
        print(f"\nDeleted {result.rowcount} manifest rows.")
        print("Next sync will re-discover and re-ingest all documents.")


if __name__ == "__main__":
    asyncio.run(main())