import argparse
import typer
import yaml
from loguru import logger
from pai_rag.ingestion.models.config.base import ParserConfig
from pai_rag.ingestion.ray_executor import RayExecutor
from pai_rag.ingestion.utils.compute_resource_utils import (
    enforce_min_requirements,
)
from llama_index.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


app = typer.Typer()


@app.command()
def parse(
    input_path: str=typer.Argument(help="The input path to the data."),
    output_path: str=typer.Argument(help="The output path to the data."),
    num_cpus: int=typer.Argument(help="Cpu required for each parse process."),
    memory: int=typer.Argument(help="Memory(GB) required for each parse process."),
    num_gpus: int=typer.Argument(default=0, help="Gpu required for each parse process."),
    enable_pdf_ocr: bool=typer.Argument(default=False, help="Whether to enable OCR for pdf files."),
    concat_sheet_rows: bool=typer.Argument(default=False, help="Whether to concat sheet rows."),
    supported_file_types_str: str=typer.Argument(default="pdf,txt,csv,xlsx,xls,docx,md,html,htm", help="The supported file extensions."),
):
    supported_file_types = [x.strip() for x in supported_file_types_str.split(",")]
    supported_file_types = [x for x in supported_file_types if x]

    parser_config = ParserConfig(
        input_path=input_path,
        output_path=output_path,
        num_cpus=num_cpus,
        memory=memory,
        num_gpus=num_gpus,
        supported_file_types=supported_file_types,
        enable_pdf_ocr=enable_pdf_ocr,
        concat_sheet_rows=concat_sheet_rows,
    )

@app.command()
def split(
    input_path: str=typer.Argument(help="The input path to the data."),
    output_path: str=typer.Argument(help="The output path to the data."),
    cpu_required: int=typer.Argument(help="Cpu required for each split process."),
    mem_required: int=typer.Argument(help="Memory(GB) required for each parse process."),
    paragraph_separator: str=typer.Argument(default=None, help="Separator between paragraphs."),
    chunk_size: int=typer.Argument(default=DEFAULT_CHUNK_SIZE, help="Chunk size for each chunk."),
    chunk_overlap: int=typer.Argument(default=DEFAULT_CHUNK_OVERLAP, help="The token overlap of each chunk when splitting."),
):
    print("split cmd")


@app.command()
def embed(
    input_path: str=typer.Argument(help="The input path to the data."),
    output_path: str=typer.Argument(help="The output path to the data."),
    cpu_required: int=typer.Argument(help="Cpu required for each split process."),
    mem_required: int=typer.Argument(help="Memory(GB) required for each parse process."),
    gpu_required: int=typer.Argument(default=0, help="Gpu required for each split process."),
    model: str=typer.Argument(default="bge-m3", help="Embedding model name."),
    connection_name: str=typer.Argument(default=None, help="Langstudio connection."),
    workspace_id: str=typer.Argument(default=None, help="PAI workspace id."),
    enable_sparse: bool=typer.Argument(default=False, help="Whether to enable sparse embedding."),
):
    print("embed cmd")


@app.command()
def e2e(
    input_path: str=typer.Argument(help="The input path to the data."),
    output_path: str=typer.Argument(help="The output path to the data."),
    config_path: str=typer.Argument(help="The path to the config file."),
):
    print("e2e cmd")
    

if __name__ == "__main__":
    app()