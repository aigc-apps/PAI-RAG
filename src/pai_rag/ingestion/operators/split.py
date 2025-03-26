from typing import List, Optional
from pai_rag.core.rag_module import resolve
from pai_rag.ingestion.operators.base import BaseOperator, OperatorName
from pai_rag.ingestion.utils.formatters import (
    convert_list_to_documents,
    convert_nodes_to_list,
)
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import (
    NodeParserConfig,
    PaiNodeParser,
)
import ray


@ray.remote
class Splitter(BaseOperator):
    def __init__(
        self,
        name: str = OperatorName.SPLITTER,
        batch_size: int = 10,
        device: str = "cpu",
        num_cpus: float = 1,
        num_gpus: Optional[float] = None,
        model_dir: str = None,
        output_filename: str = None,
        type: str = "Token",
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        **kwargs,
    ):
        super().__init__(
            name=name,
            batch_size=batch_size,
            device=device,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            model_dir=model_dir,
            output_filename=output_filename,
            **kwargs,
        )

        self.node_parser_config = NodeParserConfig(
            type=type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.node_parser = resolve(
            cls=PaiNodeParser,
            parser_config=self.node_parser_config,
        )
        self.logger.info(
            f"""SplitterActor [PaiNodeParser] init finished with following parameters:
                        type: {type}
                        chunk_size: {chunk_size}
                        chunk_overlap: {chunk_overlap}
            """
        )

    def process(self, input_docs: List[dict]) -> List[dict]:
        documents = convert_list_to_documents(input_docs)
        nodes = self.node_parser.get_nodes_from_documents(documents)
        results = convert_nodes_to_list(nodes)
        self.persist(results)
        return results
