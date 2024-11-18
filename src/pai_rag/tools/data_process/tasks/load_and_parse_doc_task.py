from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.utils.oss_client import OssClient
from pai_rag.core.rag_module import resolve
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.tools.data_process.utils.format_document import document_to_dict


def load_and_parse_doc_task(config_file, input_file):
    config = RagConfigManager.from_file(config_file).get_value()
    data_reader_config = config.data_reader
    oss_store = None
    if config.oss_store.bucket:
        oss_store = resolve(
            cls=OssClient,
            bucket_name=config.oss_store.bucket,
            endpoint=config.oss_store.endpoint,
        )
    data_reader = resolve(
        cls=PaiDataReader,
        reader_config=data_reader_config,
        oss_store=oss_store,
    )
    document = data_reader.load_single_data(input_file)
    return document_to_dict(document)
