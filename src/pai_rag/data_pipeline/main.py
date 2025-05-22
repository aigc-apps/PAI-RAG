from pathlib import Path
import dotenv
import yaml

from pai_rag.data_pipeline.utils.file_ext_utils import parse_file_extensions

dotenv.load_dotenv()


import os
import typer
from loguru import logger
from pai_rag.data_pipeline.models.config.datasource import DataSourceConfig
from pai_rag.data_pipeline.models.config.operator import (
    EmbedderConfig,
    ParserConfig,
    SplitterConfig,
    SinkConfig,
    OperatorName,
)
from pai_rag.data_pipeline.ray_executor import ray_executor
from llama_index.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

from pai_rag.integrations.embeddings.pai.pai_embedding_config import SupportedEmbedType
from pai_rag.file.nodeparsers.pai.pai_node_parser import NodeParserType
from pai_rag.utils.constants import DEFAULT_PARAGRAPH_SEP

app = typer.Typer()

DEFAULT_FILE_EXTENSIONS_STR = "pdf,txt,csv,xlsx,xls,docx,md,html,htm,jsonl"
DEFAULT_E2E_CONFIG_FILE = Path(__file__).parent / "e2e_config.yaml"


@app.command()
def data_source(
    input_path: str = typer.Option(help="The input path to the data."),
    output_path: str = typer.Option(help="The output path to the data."),
    enable_delta: bool = typer.Option(
        default=False, help="Whether to load file changes only.", show_default=True
    ),
    supported_file_types_str: str = typer.Option(
        default=DEFAULT_FILE_EXTENSIONS_STR,
        help="The supported file extensions.",
        show_default=True,
    ),
    target_index: str = typer.Option(
        default=None,
        show_default=True,
        help="The path to the index manifest or the ID of the registered dataset(DataType=INDEX) in PAI.",
    ),
    target_index_version: str = typer.Option(
        default=None,
        show_default=True,
        help="The version name of the knowledge base, used for incremental ingestion.",
    ),
    pai_rag_token: str = typer.Option(
        default=None, show_default=True, help="The PAI-RAG API key to use."
    ),
    pai_rag_endpoint: str = typer.Option(
        default=None, show_default=True, help="The PAI-RAG endpoint to use."
    ),
    pai_rag_knowledgebase: str = typer.Option(
        default=None, show_default=True, help="The knowledgebase name to use."
    ),
    pai_rag_embed_dims: str = typer.Option(
        default=1024, show_default=True, help="Default embedding dimensions."
    ),
):
    logger.info("Read execution started.")
    data_source_config = DataSourceConfig(
        input_path=input_path,
        output_path=os.path.join(output_path, OperatorName.DATA_SOURCE.value),
        enable_delta=enable_delta,
        file_extensions=parse_file_extensions(supported_file_types_str),
        target_index=target_index,
        target_index_version=target_index_version,
        pai_rag_token=pai_rag_token or os.environ.get("PAI_RAG_TOKEN"),
        pai_rag_endpoint=pai_rag_endpoint or os.environ.get("PAI_RAG_ENDPOINT"),
        pai_rag_knowledgebase=pai_rag_knowledgebase
        or os.environ.get("PAI_RAG_KNOWLEDGEBASE", "default"),
        pai_rag_embed_dims=pai_rag_embed_dims,
    )
    ray_executor.run(op_configs=[], datasource_config=data_source_config)
    logger.info("Read execution completed.")


@app.command()
def parse(
    input_path: str = typer.Option(help="The input path to the data."),
    output_path: str = typer.Option(help="The output path to the data."),
    num_cpus: int = typer.Option(
        default=1, show_default=True, help="Cpu required for each parse process."
    ),
    memory: int = typer.Option(
        default=8, show_default=True, help="Memory(GB) required for each parse process."
    ),
    num_gpus: int = typer.Option(
        default=0, show_default=True, help="Gpu required for each parse process."
    ),
    enable_pdf_ocr: bool = typer.Option(
        default=False, help="Whether to enable OCR for pdf files."
    ),
    concat_sheet_rows: bool = typer.Option(
        default=False, help="Whether to concat sheet rows."
    ),
    concurrency: int = typer.Option(
        default=1, show_default=True, help="Concurrency of sink op."
    ),
):
    logger.info("Parser execution started.")
    parser_config = ParserConfig(
        input_path=os.path.join(input_path, OperatorName.DATA_SOURCE.value),
        output_path=os.path.join(output_path, OperatorName.PARSER.value),
        num_cpus=num_cpus,
        memory=memory,
        num_gpus=num_gpus,
        enable_pdf_ocr=enable_pdf_ocr,
        concat_sheet_rows=concat_sheet_rows,
        concurrency=concurrency,
    )
    ray_executor.run(op_configs=[parser_config])
    logger.info("Parser execution completed.")


@app.command()
def split(
    input_path: str = typer.Option(help="The input path to the data."),
    output_path: str = typer.Option(help="The output path to the data."),
    num_cpus: int = typer.Option(
        default=1, show_default=True, help="Cpu required for each split process."
    ),
    memory: int = typer.Option(
        default=1, show_default=True, help="Memory(GB) required for each parse process."
    ),
    paragraph_separator: str = typer.Option(
        default=DEFAULT_PARAGRAPH_SEP,
        show_default=True,
        help="Separator between paragraphs.",
    ),
    chunk_size: int = typer.Option(
        default=DEFAULT_CHUNK_SIZE, show_default=True, help="Chunk size for each chunk."
    ),
    node_parser_type: NodeParserType = typer.Option(
        default=NodeParserType.SENTENCE, show_choices=True, help="Node parser type."
    ),
    chunk_overlap: int = typer.Option(
        default=DEFAULT_CHUNK_OVERLAP,
        show_default=True,
        help="The token overlap of each chunk when splitting.",
    ),
    concurrency: int = typer.Option(
        default=1, show_default=True, help="Concurrency of sink op."
    ),
):
    logger.info("Splitter execution started.")
    splitter_config = SplitterConfig(
        input_path=os.path.join(input_path, OperatorName.PARSER.value),
        output_path=os.path.join(output_path, OperatorName.SPLITTER.value),
        num_cpus=num_cpus,
        memory=memory,
        paragraph_separator=paragraph_separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        node_parser_type=node_parser_type,
        concurrency=concurrency,
    )
    ray_executor.run(op_configs=[splitter_config])
    logger.info("Splitter execution completed.")


@app.command()
def embed(
    input_path: str = typer.Option(help="The input path to the data."),
    output_path: str = typer.Option(help="The output path to the data."),
    num_cpus: float = typer.Option(
        default=1, show_default=True, help="Cpu required for each embedding process."
    ),
    memory: int = typer.Option(
        default=8,
        show_default=True,
        help="Memory(GB) required for each embedding process.",
    ),
    num_gpus: float = typer.Option(
        default=0, show_default=True, help="Gpu required for each embedding process."
    ),
    batch_size: int = typer.Option(
        default=32, help="batch size for embedding process."
    ),
    source: SupportedEmbedType = typer.Option(
        default="huggingface",
        show_default=True,
        help="The source of the embedding type.",
        show_choices=True,
    ),
    model: str = typer.Option(
        default="bge-m3", show_default=True, help="Embedding model name."
    ),
    connection_name: str = typer.Option(
        default=None, show_default=True, help="Langstudio connection."
    ),
    workspace_id: str = typer.Option(
        default=None, show_default=True, help="PAI workspace id."
    ),
    enable_sparse: bool = typer.Option(
        default=False, show_default=True, help="Whether to enable sparse embedding."
    ),
    concurrency: int = typer.Option(
        default=1, show_default=True, help="Concurrency of sink op."
    ),
):
    logger.info("Embedder execution started.")
    embedder_config = EmbedderConfig(
        input_path=os.path.join(input_path, OperatorName.SPLITTER.value),
        output_path=os.path.join(output_path, OperatorName.EMBEDDER.value),
        num_cpus=num_cpus,
        memory=memory,
        num_gpus=num_gpus,
        source=source,
        model=model,
        enable_sparse=enable_sparse,
        connection_name=connection_name,
        workspace_id=workspace_id,
        batch_size=batch_size,
        concurrency=concurrency,
    )
    ray_executor.run(op_configs=[embedder_config])
    logger.info("Embedder execution completed.")


@app.command()
def data_sink(
    input_path: str = typer.Option(help="The input path to the data."),
    output_path: str = typer.Option(help="The output path to the data."),
    num_cpus: int = typer.Option(
        default=1, show_default=True, help="Cpu required for each embedding process."
    ),
    memory: int = typer.Option(
        default=2,
        show_default=True,
        help="Memory(GB) required for each embedding process.",
    ),
    pai_rag_endpoint: str = typer.Option(
        default=None, help="Endpoint of PAI-RAG service."
    ),
    pai_rag_token: str = typer.Option(default=None, help="Token of PAI-RAG service."),
    batch_size: int = typer.Option(default=300, help="batch size for write process."),
    pai_rag_knowledgebase: str = typer.Option(
        default=None, show_default=True, help="Knowledgebase name to save data."
    ),
    pai_rag_embed_dims: int = typer.Option(
        default=1024, show_default=True, help="Embedding dimensions."
    ),
    concurrency: int = typer.Option(
        default=1, show_default=True, help="Concurrency of sink op."
    ),
):
    pai_rag_endpoint = pai_rag_endpoint or os.environ.get("PAI_RAG_ENDPOINT")
    pai_rag_token = pai_rag_token or os.environ.get("PAI_RAG_TOKEN")
    pai_rag_knowledgebase = pai_rag_knowledgebase or os.environ.get(
        "PAI_RAG_KNOWLEDGEBASE", "default"
    )

    assert pai_rag_endpoint, "Please provide pai_rag_endpoint to ingest into."
    assert pai_rag_token, "Please provide pai_rag_token to ingest into."

    logger.info("Write data_sink execution started.")
    sink_config = SinkConfig(
        input_path=os.path.join(input_path, OperatorName.EMBEDDER.value),
        output_path=os.path.join(output_path, OperatorName.DATA_SINK.value),
        num_cpus=num_cpus,
        memory=memory,
        pai_rag_embed_dims=pai_rag_embed_dims,
        pai_rag_token=pai_rag_token,
        pai_rag_endpoint=pai_rag_endpoint,
        pai_rag_knowledgebase=pai_rag_knowledgebase,
        concurrency=concurrency,
        batch_size=batch_size,
    )
    ray_executor.run(op_configs=[sink_config])
    logger.info("Write data_sink execution completed.")


@app.command()
def e2e(
    input_path: str = typer.Option(help="The input path to the data."),
    output_path: str = typer.Option(help="The output path to the data."),
    config_file: str = typer.Option(
        default=DEFAULT_E2E_CONFIG_FILE,
        show_default=True,
        help="The path to the config file.",
    ),
):
    read_output_path = os.path.join(output_path, OperatorName.DATA_SOURCE.value)
    parse_output_path = os.path.join(output_path, OperatorName.PARSER.value)
    split_output_path = os.path.join(output_path, OperatorName.SPLITTER.value)
    embed_output_path = os.path.join(output_path, OperatorName.EMBEDDER.value)
    write_output_path = os.path.join(output_path, OperatorName.DATA_SINK.value)

    with open(config_file) as file_handler:
        e2e_yaml = yaml.safe_load(file_handler)
        operators_yaml = e2e_yaml["operators"]
        op_yaml_map = {}
        for op_yaml in operators_yaml:
            op_yaml_map[op_yaml["name"]] = op_yaml

        assert (
            "data_source" in op_yaml_map
        ), "data_source op is required for e2e pipeline."
        datasource_yaml = op_yaml_map["data_source"]

        supported_file_types_str = datasource_yaml.get(
            "supported_file_types_str", DEFAULT_FILE_EXTENSIONS_STR
        )
        datasource_config = DataSourceConfig(
            input_path=input_path,
            output_path=read_output_path,
            enable_delta=datasource_yaml.get("enable_delta", True),
            file_extensions=parse_file_extensions(supported_file_types_str),
            target_index=datasource_yaml.get("target_index"),
            target_index_version=datasource_yaml.get("target_index_version"),
            pai_rag_endpoint=datasource_yaml.get("pai_rag_endpoint")
            or os.environ.get("PAI_RAG_ENDPOINT"),
            pai_rag_token=datasource_yaml.get("pai_rag_token")
            or os.environ.get("PAI_RAG_TOKEN"),
            pai_rag_knowledgebase=datasource_yaml.get("pai_rag_knowledgebase")
            or os.environ.get("PAI_RAG_KNOWLEDGEBASE", "default"),
            pai_rag_embed_dims=datasource_yaml.get("pai_rag_embed_dims", 1024),
        )

        assert "parse" in op_yaml_map, "parse op is required for e2e pipeline."
        parse_yaml = op_yaml_map["parse"]
        parse_config = ParserConfig(
            input_path=read_output_path,
            output_path=parse_output_path,
            num_cpus=parse_yaml.get("num_cpus", 1),
            memory=parse_yaml.get("memory", 8),
            num_gpus=parse_yaml.get("num_gpus", 0),
            enable_pdf_ocr=parse_yaml.get("enable_pdf_ocr", False),
            concat_sheet_rows=parse_yaml.get("concat_sheet_rows", False),
            concurrency=parse_yaml.get("concurrency", 1),
        )

        assert "split" in op_yaml_map, "split op is required for e2e pipeline."
        split_yaml = op_yaml_map["split"]
        split_config = SplitterConfig(
            input_path=parse_output_path,
            output_path=split_output_path,
            num_cpus=split_yaml.get("num_cpus", 1),
            memory=split_yaml.get("memory", 4),
            num_gpus=split_yaml.get("num_gpus", 0),
            chunk_size=split_yaml.get("chunk_size", DEFAULT_CHUNK_SIZE),
            chunk_overlap=split_yaml.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            node_parser_type=split_yaml.get("node_parser_type", NodeParserType.TOKEN),
            paragraph_separator=split_yaml.get(
                "paragraph_separator", DEFAULT_PARAGRAPH_SEP
            ),
            concurrency=split_yaml.get("concurrency", 1),
        )

        assert "embed" in op_yaml_map, "embed op is required for e2e pipeline."
        embed_yaml = op_yaml_map["embed"]
        embed_config = EmbedderConfig(
            input_path=split_output_path,
            output_path=embed_output_path,
            model=embed_yaml.get("model", "bge-m3"),
            source=embed_yaml.get("source", "huggingface"),
            batch_size=embed_yaml.get("batch_size", 32),
            concurrency=embed_yaml.get("concurrency", 1),
            num_cpus=4,
            num_gpus=1,
            memory=12,
        )

        assert "data_sink" in op_yaml_map, "data_sink op is required for e2e pipeline."
        sink_yaml = op_yaml_map["data_sink"]
        sink_config = SinkConfig(
            input_path=embed_output_path,
            output_path=write_output_path,
            pai_rag_endpoint=sink_yaml.get("pai_rag_endpoint")
            or os.environ.get("PAI_RAG_ENDPOINT"),
            pai_rag_token=sink_yaml.get("pai_rag_token")
            or os.environ.get("PAI_RAG_TOKEN"),
            pai_rag_knowledgebase=sink_yaml.get("pai_rag_knowledgebase")
            or os.environ.get("PAI_RAG_KNOWLEDGEBASE", "default"),
            pai_rag_embed_dims=sink_yaml.get("pai_rag_embed_dims", 1024),
            concurrency=sink_yaml.get("concurrency", 1),
            num_cpus=sink_yaml.get("num_cpus", 1),
            num_gpus=sink_yaml.get("num_gpus", 0),
            memory=sink_yaml.get("memory", "4GB"),
            batch_size=sink_yaml.get("batch_size", 100),
        )

        logger.info("Starting e2e execution...")
        ray_executor.run(
            datasource_config=datasource_config,
            op_configs=[parse_config, split_config, embed_config, sink_config],
        )
        logger.info("Finished e2e execution...")


if __name__ == "__main__":
    app()
