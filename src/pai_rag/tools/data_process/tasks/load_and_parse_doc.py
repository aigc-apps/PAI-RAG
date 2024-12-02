import os
from pai_rag.integrations.readers.pai.pai_data_reader import PaiDataReader
from pai_rag.utils.oss_client import OssClient
from pai_rag.core.rag_module import resolve
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.tools.data_process.utils.format_document import document_to_dict
from pai_rag.utils.download_models import ModelScopeDownloader

RAY_ENV_MODEL_DIR = "/PAI-RAG/pai_rag_model_repository"
os.environ["PAI_RAG_MODEL_DIR"] = RAY_ENV_MODEL_DIR


def load_and_parse_doc_task(config_file, input_file):
    config = RagConfigManager.from_file(config_file).get_value()
    download_models = ModelScopeDownloader(
        fetch_config=True, download_directory_path=RAY_ENV_MODEL_DIR
    )
    download_models.load_mineru_config()
    download_models.load_models(model="PDF-Extract-Kit")

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
    documents = data_reader.load_data(file_path_or_directory=input_file)
    return document_to_dict(documents[0])
