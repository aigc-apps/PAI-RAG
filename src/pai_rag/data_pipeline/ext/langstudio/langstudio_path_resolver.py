import json
import os
import posixpath
from typing import List

from loguru import logger
from pai_rag.data_pipeline.ext.langstudio.langstudio_client import (
    get_dlc_client,
    get_workspace_client,
)
from pai_rag.data_pipeline.ext.langstudio.langstudio_constants import (
    WORKSPACE_ID_FROM_ENV,
)
from pai_rag.data_pipeline.ext.langstudio.oss_utils import standardize_oss_uri
from pai_rag.data_pipeline.utils.path_resolver import MountPathResolver
from alibabacloud_pai_dlc20201203.models import GetJobRequest


import dataclasses


@dataclasses.dataclass
class DataMountConfig:
    mount_point: str  # 目标挂载路径
    source_uri: str  # 源路径

    def __post_init__(self):
        # ensure no trailing slash
        self.mount_point = self.mount_point.rstrip("/")
        self.data_source_path = self.data_source_path.rstrip("/")
        self.data_source_path = standardize_oss_uri(self.data_source_path)


@classmethod
def get_mount_path_from_dataset_options(dataset_options: str) -> str:
    try:
        dataset = json.loads(dataset_options)
        mount_path = dataset.get("mountPath")
    except Exception as e:
        logger.warning(f"Invalid dataset options: {dataset_options}, error: {e}")
        mount_path = None
    return mount_path


class LangStudioPathResolver(MountPathResolver):
    def __init__(
        self,
        workspace_id: str,
        mount_configs: List[DataMountConfig] = [],
    ):
        super().__init__()
        self.workspace_id = workspace_id
        self.mount_configs = mount_configs
        logger.info(
            f"LangStudioPathResolver init with workspace_id: {workspace_id} and {mount_configs}"
        )

    @classmethod
    def from_env(cls):
        dlc_job_id = os.getenv("DLC_JOB_ID")
        workspace_id = WORKSPACE_ID_FROM_ENV

        if not dlc_job_id or not workspace_id:
            raise RuntimeError(
                "Not running in PAI DLC Job environment, 'PAI_WORKSPACE_ID' or 'DLC_JOB_ID' is not set"
            )

        dlc_client = get_dlc_client()
        ws_client = get_workspace_client()

        resp = dlc_client.get_job(job_id=dlc_job_id, request=GetJobRequest())
        job_data_sources = resp.body.data_sources
        data_mount_configs = []
        for ds in job_data_sources:
            if not ds.uri and not ds.data_source_id:
                raise RuntimeError("Invalid job data source: {}".format(ds))
            # Get DataSource URI
            if not ds.uri:
                dataset = ws_client.get_dataset(dataset_id=ds.data_source_id).body
                ds.uri = dataset.uri
                if not ds.mount_path:
                    # get the default mount path from dataset.options
                    ds.mount_path = get_mount_path_from_dataset_options(dataset.options)

            if not ds.uri.startswith("oss://"):
                raise RuntimeError("Unsupported data source URI: {}".format(ds.uri))

            data_mount_configs.append(
                DataMountConfig(
                    mount_point=ds.mount_path,
                    source_uri=ds.uri,
                )
            )

        return cls(
            workspace_id=workspace_id,
            data_mount_configs=data_mount_configs,
        )

    def resolve_destination_path(self, uri: str) -> str:
        """
        Resolve the path to the local file system.
        """
        if not uri.startswith("oss://"):
            raise ValueError(f"Unsupported data source URI: {uri}")

        for config in self.mount_configs:
            if uri.startswith(config.source_uri):
                relative_path = posixpath.relpath(uri, config.source_uri)
                if relative_path == ".":
                    relative_path = ""
                return os.path.join(config.mount_point, relative_path)
        raise ValueError(f"Path '{uri}' is not in any data mount point.")

    def resolve_source_url(self, path: str) -> str:
        """
        Resolve the path to the local file system.
        """
        if path.startswith("oss://"):
            return path

        if path.startswith("nas://"):
            raise RuntimeError("Not support NAS data source")

        full_path = os.path.abspath(path)
        for mount_config in self.mount_configs:
            if (
                posixpath.commonpath([full_path, mount_config.mount_point])
                == mount_config.mount_point
            ):
                relative_path = posixpath.relpath(full_path, mount_config.mount_point)
                if relative_path == ".":
                    relative_path = ""
                return posixpath.join(mount_config.source_uri, relative_path)

        logger.warning(f"Path {path} does not match any datasource mount point")
        return path
