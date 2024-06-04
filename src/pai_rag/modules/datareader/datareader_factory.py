from typing import Any, Dict, List
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.integrations.readers.pai_pdf_reader import PaiPDFReader
from pai_rag.integrations.readers.llama_parse_reader import LlamaParseDirectoryReader
from pai_rag.integrations.readers.html.html_reader import HtmlReader
from pai_rag.integrations.readers.pai_csv_reader import PaiPandasCSVReader
from pai_rag.integrations.readers.pai_excel_reader import PaiPandasExcelReader
from llama_index.readers.database import DatabaseReader
from llama_index.core import SimpleDirectoryReader
import logging

logger = logging.getLogger(__name__)


class DataReaderFactoryModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.reader_config = new_params[MODULE_PARAM_CONFIG]
        self.file_readers = {
            ".html": HtmlReader(),
            ".htm": HtmlReader(),
            ".pdf": PaiPDFReader(
                enable_image_ocr=self.reader_config.get("enable_image_ocr", False),
                model_dir=self.reader_config.get("easyocr_model_dir", None),
            ),
            ".csv": PaiPandasCSVReader(
                concat_rows=self.reader_config.get("concat_rows", False),
            ),
            ".xlsx": PaiPandasExcelReader(
                concat_rows=self.reader_config.get("concat_rows", False),
            ),
            ".xls": PaiPandasExcelReader(
                concat_rows=self.reader_config.get("concat_rows", False),
            ),
        }
        return self

    def get_reader(self, directory: str):
        if self.reader_config["type"] == "SimpleDirectoryReader":
            return SimpleDirectoryReader(
                input_dir=directory,
                file_extractor=self.file_readers,
            )

        elif self.reader_config["type"] == "LlamaParseDirectoryReader":
            return LlamaParseDirectoryReader(
                input_dir=directory,
                api_key=self.reader_config["llama_cloud_api_key"],
                oss_cache=self.oss_cache,
            )
        elif (
            self.reader_config["type"] == "DatabaseReader"
            and self.reader_config["database_type"] == "PostgreSQL"
        ):
            logger.info(f"Loaded DatabaseReader with {self.reader_config}.")

            return DatabaseReader(
                scheme="postgresql+psycopg2",  # can also support other databases like Mysql using 'mysql+pymysql'
                host=self.reader_config["host"],
                port=self.reader_config["port"],
                dbname=self.reader_config["dbname"],
                user=self.reader_config["user"],
                password=self.reader_config["password"],
            )
        else:
            error_msg = f"Unknown data reader type '{self.reader_config['name']}'."
            logger.error(error_msg)
            raise ValueError(error_msg)
