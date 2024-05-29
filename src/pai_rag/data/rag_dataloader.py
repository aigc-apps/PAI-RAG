from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM

from pai_rag.utils.store_utils import store_path
from pai_rag.integrations.extractors.html_qa_extractor import HtmlQAExtractor
from pai_rag.integrations.extractors.text_qa_extractor import TextQAExtractor
from pai_rag.modules.nodeparser.node_parser import node_id_hash

import logging

logger = logging.getLogger(__name__)


DEFAULT_LOCAL_QA_MODEL_PATH = "/huggingface/transformers/qwen_1.8b"


class RagDataLoader:
    """
    RagDataLoader:
    Load data with corresponding data readers according to config.
    """

    def __init__(
        self,
        datareader_factory,
        node_parser,
        index,
        oss_cache,
        use_local_qa_model=False,
    ):
        self.datareader_factory = datareader_factory
        self.node_parser = node_parser
        self.index = index
        self.oss_cache = oss_cache

        if use_local_qa_model:
            # API暂不支持此选项
            self.qa_llm = HuggingFaceLLM(
                model_name=DEFAULT_LOCAL_QA_MODEL_PATH,
                tokenizer_name=DEFAULT_LOCAL_QA_MODEL_PATH,
            )
        else:
            self.qa_llm = Settings.llm
        html_extractor = HtmlQAExtractor(llm=self.qa_llm)
        txt_extractor = TextQAExtractor(llm=self.qa_llm)

        self.extractors = [html_extractor, txt_extractor]

        logger.info("RagDataLoader initialized.")

    async def load(self, file_directory: str, enable_qa_extraction: bool):
        data_reader = self.datareader_factory.get_reader(file_directory)
        docs = data_reader.load_data()
        nodes = []
        for doc in docs:
            if doc.metadata.get("file_type", "Unknown") == "HTML":
                node_id = node_id_hash(0, doc)
                nodes.append(
                    TextNode(id_=node_id, text=doc.text, metadata=doc.metadata)
                )
            else:
                nodes.extend(self.node_parser.get_nodes_from_documents([doc]))

        # QA metadata extraction
        if enable_qa_extraction:
            qa_nodes = []

            for extractor in self.extractors:
                metadata_list = await extractor.aextract(nodes)
                for i, node in enumerate(nodes):
                    qa_extraction_result = metadata_list[i].get(
                        "qa_extraction_result", {}
                    )
                    q_cnt = 0
                    metadata = node.metadata
                    for q, a in qa_extraction_result.items():
                        metadata["answer"] = a
                        qa_nodes.append(
                            TextNode(
                                id_=f"{node.id_}_{q_cnt}", text=q, metadata=metadata
                            )
                        )
                        q_cnt += 1
            for node in qa_nodes:
                node.excluded_embed_metadata_keys.append("answer")
                node.excluded_llm_metadata_keys.append("question")
            nodes.extend(qa_nodes)

        self.index.insert_nodes(nodes)
        self.index.storage_context.persist(persist_dir=store_path.persist_path)
        logger.info(f"Inserted {len(nodes)} nodes successfully.")
        return
