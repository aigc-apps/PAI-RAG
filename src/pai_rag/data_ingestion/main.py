import typer
from loguru import logger
from pai_rag.data_ingestion.models.config.datasource import DataSourceConfig
from pai_rag.data_ingestion.models.config.operator import EmbedderConfig, ParserConfig, SplitterConfig, WriterConfig
from pai_rag.data_ingestion.ray_executor import ray_executor
from pai_rag.data_ingestion.utils.compute_resource_utils import (
    enforce_min_requirements,
)
from llama_index.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

from pai_rag.integrations.embeddings.pai.pai_embedding_config import SupportedEmbedType
from pai_rag.integrations.nodeparsers.pai.pai_node_parser import NodeParserType
from pai_rag.utils.constants import DEFAULT_PARAGRAPH_SEP


app = typer.Typer()


@app.command()
def read(
    input_path: str=typer.Option(help="The input path to the data."),
    output_path: str=typer.Option(help="The output path to the data."),
    enable_delta: bool=typer.Option(default=False, help="Whether to load file changes only.", show_default=True),
    supported_file_types_str: str=typer.Option(default="pdf,txt,csv,xlsx,xls,docx,md,html,htm", help="The supported file extensions.", show_default=True),
    target_index: str=typer.Option(help="The path to the index manifest or the ID of the registered dataset(DataType=INDEX) in PAI."),
    target_index_version: str=typer.Option(help="The version name of the knowledge base, used for incremental ingestion."),
    rag_api_key: str=typer.Option(help="The RAG API key to use."),
    rag_endpoint: str=typer.Option(help="The RAG endpoint to use."),
):
    logger.info("Read execution started.")
    data_source_config = DataSourceConfig(
        input_path=input_path,
        output_path=output_path,
        enable_delta=enable_delta,
        file_extensions=supported_file_types_str.split(","),
        target_index=target_index,
        target_index_version=target_index_version,
        rag_api_key=rag_api_key,
        rag_endpoint=rag_endpoint,
    )
    ray_executor.run(op_configs=[], datasource_config=data_source_config)
    logger.info("Read execution completed.")


@app.command()
def parse(
    input_path: str=typer.Option(help="The input path to the data."),
    output_path: str=typer.Option(help="The output path to the data."),
    num_cpus: int=typer.Option(help="Cpu required for each parse process."),
    memory: int=typer.Option(help="Memory(GB) required for each parse process."),
    num_gpus: int=typer.Option(default=0, help="Gpu required for each parse process."),
    enable_pdf_ocr: bool=typer.Option(default=False, help="Whether to enable OCR for pdf files."),
    concat_sheet_rows: bool=typer.Option(default=False, help="Whether to concat sheet rows."),
):
    logger.info("Parser execution started.")
    parser_config = ParserConfig(
        input_path=input_path,
        output_path=output_path,
        num_cpus=num_cpus,
        memory=memory,
        num_gpus=num_gpus,
        enable_pdf_ocr=enable_pdf_ocr,
        concat_sheet_rows=concat_sheet_rows,
    )
    ray_executor.run(op_configs=[parser_config])
    logger.info("Parser execution completed.")

@app.command()
def split(
    input_path: str=typer.Option(help="The input path to the data."),
    output_path: str=typer.Option(help="The output path to the data."),
    num_cpus: int=typer.Option(help="Cpu required for each split process."),
    memory: int=typer.Option(help="Memory(GB) required for each parse process."),
    paragraph_separator: str=typer.Option(default=DEFAULT_PARAGRAPH_SEP, help="Separator between paragraphs."),
    chunk_size: int=typer.Option(default=DEFAULT_CHUNK_SIZE, help="Chunk size for each chunk."),
    node_parser_type: NodeParserType=typer.Option(default=NodeParserType.TOKEN, show_choices=True, help="Node parser type."),
    chunk_overlap: int=typer.Option(default=DEFAULT_CHUNK_OVERLAP, help="The token overlap of each chunk when splitting."),
):
    logger.info("Splitter execution started.")
    splitter_config = SplitterConfig(
        input_path=input_path,
        output_path=output_path,
        num_cpus=num_cpus,
        memory=memory,
        paragraph_separator=paragraph_separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        node_parser_type=node_parser_type,
    )
    ray_executor.run(op_configs=[splitter_config])
    logger.info("Splitter execution completed.")


@app.command()
def embed(
    input_path: str=typer.Option(help="The input path to the data."),
    output_path: str=typer.Option(help="The output path to the data."),
    num_cpus: int=typer.Option(help="Cpu required for each embedding process."),
    memory: int=typer.Option(help="Memory(GB) required for each embedding process."),
    num_gpus: int=typer.Option(default=0, help="Gpu required for each embedding process."),
    batch_size: int=typer.Option(default=32, help="batch size for embedding process."),
    source: SupportedEmbedType= typer.Option(help="The source of the embedding type.", show_choices=True),
    model: str=typer.Option(default="bge-m3", help="Embedding model name."),
    connection_name: str=typer.Option(default=None, help="Langstudio connection."),
    workspace_id: str=typer.Option(default=None, help="PAI workspace id."),
    enable_sparse: bool=typer.Option(default=False, help="Whether to enable sparse embedding."),
):
    logger.info("Embedder execution started.")
    embedder_config = EmbedderConfig(
        input_path=input_path,
        output_path=output_path,
        num_cpus=num_cpus,
        memory=memory,
        num_gpus=num_gpus,
        source=source,
        model=model,
        enable_sparse=enable_sparse,
        connection_name=connection_name,
        workspace_id=workspace_id,
        batch_size=batch_size,
    )
    ray_executor.run(op_configs=[embedder_config])
    logger.info("Embedder execution completed.")


@app.command()
def write(
    input_path: str=typer.Option(help="The input path to the data."),
    num_cpus: int=typer.Option(help="Cpu required for each embedding process."),
    memory: int=typer.Option(help="Memory(GB) required for each embedding process."),
    rag_endpoint: str=typer.Option(help="Endpoint of PAI-RAG service."),
    rag_key: str=typer.Option(help="Token of PAI-RAG service."),
    knowledgebase: str=typer.Option(default="default", show_default=True, help="Knowledgebase name to save data."),
    embed_dims: int=typer.Option(default=1024, show_default=True, help="Embedding dimensions."),
):
    logger.info("Writer execution started.")
    writer_config = WriterConfig(
        input_path=input_path,
        output_path="dummy",
        num_cpus=num_cpus,
        memory=memory,
        knowledgebase=knowledgebase,
        embed_dims=embed_dims,
        rag_endpoint=rag_endpoint,
        rag_key=rag_key,
    )
    ray_executor.run(op_configs=[writer_config])
    logger.info("Writer execution completed.")


@app.command()
def e2e(
    input_path: str=typer.Option(help="The input path to the data."),
    output_path: str=typer.Option(help="The output path to the data."),
    config_path: str=typer.Option(help="The path to the config file."),
):
    raise NotImplementedError("E2E is not implemented yet.")
    

if __name__ == "__main__":
    app()