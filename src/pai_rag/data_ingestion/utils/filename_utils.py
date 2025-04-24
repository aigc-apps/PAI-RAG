from ray.data.datasource import FilenameProvider

class BlockFileNameProvider(FilenameProvider):
    def __init__(self, run_label: str, file_format: str):
        self._run_label = run_label
        self._file_format = file_format

    def get_filename_for_block(self, block, task_index, block_index):
        return f"{self._run_label}_{task_index:06}_{block_index:06}.{self._file_format}"
