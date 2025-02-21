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

operator = "rag_splitter"
working_dir = BASE_DIR
dataset_path = os.path.join(BASE_DIR, "tests/data_process/output/rag_parser")
export_path = os.path.join(BASE_DIR, "tests/data_process/output")


def test_run_single_splitter_op():
    from pai_rag.tools.data_process.ops.splitter_op import Splitter
    from pai_rag.tools.data_process.dataset.ray_dataset import RayDataset

    splitter = Splitter.remote(
        op_name=operator,
        working_dir=str(working_dir),
        cpu_required=2,
        mem_required="2GB",
        chunk_size=200,
    )
    # 检查能否序列化
    try:
        pickle.dumps(splitter)  # 对 splitter 进行序列化测试
    except Exception as e:
        print(f"Serialization Error: {e}")
    dataset = RayDataset(str(dataset_path), cfg=Namespace(export_path=str(export_path)))
    dataset.process(
        operators=[splitter],
        op_name=operator,
    )
    splitter_export_path = Path(os.path.join(export_path, "rag_splitter"))

    # 遍历导出目录下的所有文件
    for file_path in splitter_export_path.iterdir():
        if file_path.is_file():
            with open(file_path, "r") as file:
                assert len(file.readlines()) == 9
