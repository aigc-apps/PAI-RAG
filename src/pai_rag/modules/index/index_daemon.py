import asyncio
import datetime
import os
import json
from pai_rag.modules.module_registry import module_registry
from pai_rag.modules.index.index_entry import index_entry
import logging

logger = logging.getLogger(__name__)


class IndexDaemon:
    def _get_index_lastUpdated(self, index_path):
        index_metadata_file = os.path.join(index_path, "index.metadata")
        if os.path.exists(index_metadata_file):
            with open(index_metadata_file) as f:
                metadata = json.loads(f.read())
                return metadata["lastUpdated"]
        return None

    def register(self, index_path):
        if index_path not in index_entry.index_entries:
            index_entry.index_entries[index_path] = self._get_index_lastUpdated(
                index_path
            )

    async def refresh_async(self):
        while True:
            logger.debug(f"{datetime.datetime.now()} Start scan.")
            bm25_indexes = list(
                module_registry.get_mod_instances("BM25IndexModule").values()
            )

            index_map = module_registry.get_mod_instances("IndexModule")
            for _, index in index_map.items():
                index_path = index.persist_path
                if index_path not in index_entry.index_entries:
                    continue

                lastUpdated = self._get_index_lastUpdated(index_path)
                logging.debug(
                    f"Comparing {lastUpdated} <---> {index_entry.index_entries[index_path]}"
                )
                if lastUpdated != index_entry.index_entries[index_path]:
                    logger.info(f"{datetime.datetime.now()} Reloading index.")

                    index.reload()

                    for bm25_index in bm25_indexes:
                        if bm25_index and bm25_index.persist_path == index_path:
                            logger.info(
                                f"{datetime.datetime.now()} Reloading bm25 index."
                            )
                            bm25_index.reload()

                    index_entry.index_entries[index_path] = lastUpdated
                    logger.info(f"{datetime.datetime.now()} Reloaded index.")
                    module_registry.destroy_config_cache()

            logger.debug(f"{datetime.datetime.now()} Index scan complete.")
            await asyncio.sleep(10)


index_daemon = IndexDaemon()
