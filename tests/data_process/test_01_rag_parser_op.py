import os
from pathlib import Path
from argparse import Namespace
import pickle
import pytest

BASE_DIR = Path(__file__).parent.parent.parent

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_GPU_TESTS", "false") == "true",
    reason="Need to execute in a CUDA environment.",
)

operator = "rag_parser"
working_dir = BASE_DIR
dataset_path = os.path.join(BASE_DIR, "tests/data_process/input")
export_path = os.path.join(BASE_DIR, "tests/data_process/output")


def test_run_single_parser_op():
    from pai_rag.tools.data_process.ops.parser_op import Parser
    from pai_rag.tools.data_process.dataset.file_dataset import FileDataset

    parser = Parser.remote(
        op_name=operator,
        working_dir=str(working_dir),
        cpu_required=4,
        mem_required="8GB",
    )
    # 检查能否序列化
    try:
        pickle.dumps(parser)  # 对 parser 进行序列化测试
    except Exception as e:
        print(f"Serialization Error: {e}")
    dataset = FileDataset(
        str(dataset_path), cfg=Namespace(export_path=str(export_path))
    )
    dataset.process(
        operators=[parser],
        op_name=operator,
    )
    parser_export_path = Path(os.path.join(export_path, "rag_parser"))

    # 遍历导出目录下的所有文件
    for file_path in parser_export_path.iterdir():
        if file_path.is_file():
            with open(file_path, "r") as file:
                assert len(file.readlines()) == 3
