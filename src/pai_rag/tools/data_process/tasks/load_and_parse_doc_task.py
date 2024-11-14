from pai_rag.tools.data_process.tasks.base_task import BaseTask
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.utils.oss_client import OssClient
from pai_rag.core.rag_module import resolve


class LoadAndParseDocTask(BaseTask):
    _accelerator = "cpu"

    def __init__(self, pattern=None, **kwargs):
        super().__init__(**kwargs)

        self.data_reader_config = self.config.data_reader
        self.oss_store = None
        if self.config.oss_store.bucket:
            self.oss_store = resolve(
                cls=OssClient,
                bucket_name=self.config.oss_store.bucket,
                endpoint=self.config.oss_store.endpoint,
            )
        self.pattern = pattern
        self.data_reader = resolve(
            cls=PaiDataReader,
            reader_config=self.data_reader_config,
            oss_store=self.oss_store,
        )

    def process(self, input_file):
        document = self.data_reader.load_single_data(input_file)
        return document
