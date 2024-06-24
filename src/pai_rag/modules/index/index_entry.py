import os
import json
import logging

logger = logging.getLogger(__name__)


class IndexEntry:
    def __init__(self):
        self.index_entries = {}

    def register(self, index_path):
        if index_path not in self.index_entries:
            self.index_entries[index_path] = self._get_index_lastUpdated(index_path)

    def _get_index_lastUpdated(self, index_path):
        index_metadata_file = os.path.join(index_path, "index.metadata")
        if os.path.exists(index_metadata_file):
            with open(index_metadata_file) as f:
                metadata = json.loads(f.read())
                return metadata["lastUpdated"]
        return None


index_entry = IndexEntry()
