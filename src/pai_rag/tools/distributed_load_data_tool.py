import click
import os
import json
import torch.distributed as dist
from torch.multiprocessing import spawn

_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])


def init_process():
    print(f'init_process: rank={_rank}, world_size={_world_size}")')
    dist.init_process_group(backend="nccl", rank=_rank, world_size=_world_size)


def process_file(file_path):
    # 这里添加处理文件的逻辑
    # 例如：读取文件，进行模型推理等
    with open(file_path, "r") as f:
        data = f.read()
        # 模型推理等操作
        result = data

    # 返回处理结果或状态
    return f"Processed {file_path}: \n {result}"


def update_status(file_path, status):
    print(f"[master] update_status for {file_path}: {status}")
    status_info = {"file": file_path, "status": status}

    # 你可以选择将状态写入文件或数据库
    with open("./localdata/distributed_status.json", "a") as status_file:
        json.dump(status_info, status_file)
        status_file.write("\n")


def worker(file_list):
    for file_path in file_list:
        print(f"[worker] process_file for {file_path}")
        processed_result = process_file(file_path)
        if _rank == 0:  # 只有主节点负责记录状态
            update_status(file_path, "Completed" + processed_result)


@click.command()
@click.option(
    "-d",
    "--data_path",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="data path (file or directory) to ingest.",
)
def run(data_path=None):
    init_process()
    print(f"data_path: {data_path}")
    file_list = [f for f in data_path if os.path.isfile(f)]
    print(f"Loading files count: {len(file_list)}")
    # 启动多进程
    spawn(worker, args=(_world_size, file_list), nprocs=_world_size)
