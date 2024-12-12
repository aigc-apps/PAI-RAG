import os
import ray
from pai_rag.tools.data_process.ops.base_op import BaseOP, OPERATORS
from pai_rag.tools.data_process.utils.formatters import convert_document_to_dict
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.utils.constants import DEFAULT_MODEL_DIR

OP_NAME = "pai_rag_parser"


@OPERATORS.register_module(OP_NAME)
@ray.remote()
class Parser(BaseOP):
    """Mapper to generate samples whose captions are generated based on
    another model and the figure."""

    _accelerator = "cuda"
    _batched_op = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.download_model_list = ["PDF-Extract-Kit"]
        self.load_models(self.download_model_list, self.accelerator)

    def load_models(self, model_list, accelerator):
        download_models = ModelScopeDownloader(
            fetch_config=True,
            download_directory_path=os.getenv("PAI_RAG_MODEL_DIR", DEFAULT_MODEL_DIR),
        )
        for model_name in model_list:
            download_models.load_model(model=model_name)
        download_models.load_mineru_config(accelerator)

    def process(self, input_file):
        documents = self.data_reader.load_data(file_path_or_directory=input_file)
        return convert_document_to_dict(documents[0])
