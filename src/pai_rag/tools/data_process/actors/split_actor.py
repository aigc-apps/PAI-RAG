import os
from loguru import logger
from pai_rag.core.rag_module import resolve
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import PaiNodeParser
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.tools.data_process.utils.format_document import convert_dict_to_documents
from pai_rag.tools.data_process.utils.format_node import convert_nodes_to_dict


class SplitActor:
    def __init__(self, working_dir, config_file):
        RAY_ENV_MODEL_DIR = os.path.join(working_dir, "model_repository")
        os.environ["PAI_RAG_MODEL_DIR"] = RAY_ENV_MODEL_DIR
        logger.info(f"Init SplitActor with working dir: {RAY_ENV_MODEL_DIR}.")
        config = RagConfigManager.from_file(config_file).get_value()
        self.node_parser = resolve(cls=PaiNodeParser, parser_config=config.node_parser)
        logger.info("SplitActor init finished.")

    def __call__(self, documents):
        format_documents = convert_dict_to_documents(documents)
        nodes = self.node_parser.get_nodes_from_documents(format_documents)
        return convert_nodes_to_dict(nodes)
