import os
from ray.data.datasource import ReadTask, FileBasedDatasource
from ray.air.util.tensor_extensions.arrow import pyarrow_table_from_pydict
from ray.data.block import BlockMetadata

from typing import List, Dict, Optional, Union

from pai_rag.integrations.data_analysis.nl2pandas_retriever import read_file



class FileDeltaDatasource(FileBasedDatasource):
    def __init__(
        self,
        paths: Union[str, List[str]],
        file_extensions: Optional[List[str]] = None,
    ):
        super().__init__(
            paths=paths,
            file_extensions=file_extensions,
        )

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        """
        获取读取任务，返回一个或多个 ReadTask。
        """
        def read_files() -> List[Dict]:
            """
            准备读取任务，返回一个或多个 ReadTask。
            """
            # 获取文件夹中的所有文件信息
            files_info = []
            for file_path in self._paths():
                file_name = os.path.basename(file_path)
                value_dict = {
                    "file_path": [file_path],
                    "file_name": [file_name],
                    "operation": ["add"]
                }
                files_info.append(pyarrow_table_from_pydict(value_dict))

            # 将文件信息分片为多个任务（这里简单分为一个任务）
            return files_info


        return [ReadTask(read_files, metadata=BlockMetadata(None,None,None,None,None))]
    
    def estimate_inmemory_data_size(self):
        return None
