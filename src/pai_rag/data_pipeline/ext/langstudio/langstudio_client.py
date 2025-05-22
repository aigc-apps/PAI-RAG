import json
import os
import socket
from typing import Optional
from alibabacloud_pai_dlc20201203.client import Client as DLCJobClient
from alibabacloud_pailangstudio20240710.client import Client as LangStudioClient
from alibabacloud_aiworkspace20210204.client import Client as AIWorkSpaceClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_credentials.exceptions import CredentialException
from alibabacloud_credentials.models import Config as CredentialConfig
from alibabacloud_credentials.models import CredentialModel
from alibabacloud_credentials.utils import auth_constant

from pai_rag.data_pipeline.ext.langstudio.langstudio_constants import (
    REGION_ID_FROM_ENV,
)


def is_reachable(endpoint: str, port: int = 80, timeout: int = 1) -> bool:
    """Check if the domain is connectable."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Set the timeout for the socket
    sock.settimeout(timeout)
    try:
        # Get the IP address of the domain
        ip = socket.gethostbyname(endpoint)
        # Try to connect to the IP address on specific port (default 80, HTTP)
        sock.connect((ip, port))
        # If the connection is successful, return True
        return True
    except (socket.timeout, socket.gaierror, socket.error):
        # If there is an error connecting, return False
        return False
    finally:
        sock.close()


class CustomCredentialClient(CredentialClient):
    """

    A custom credential client for EAS.

    """

    DEFAULT_EAS_CREDENTIALS = "/etc/ram/credentials"

    def __init__(self):
        try:
            self._cred_client = CredentialClient(config=CredentialConfig())
        except CredentialException as e:
            if not os.path.exists(CustomCredentialClient.DEFAULT_EAS_CREDENTIALS):
                raise e
            self._cred_client = None

    def get_credential(self) -> CredentialModel:
        if self._cred_client:
            return self._cred_client.get_credential()
        else:
            return CustomCredentialClient._get_eas_credential()

    @staticmethod
    def _get_eas_credential() -> Optional[CredentialModel]:
        if not os.path.exists(CustomCredentialClient.DEFAULT_EAS_CREDENTIALS):
            return

        with open(CustomCredentialClient.DEFAULT_EAS_CREDENTIALS, "r") as f:
            cred_json = json.load(f)
        return CredentialModel(
            access_key_id=cred_json["AccessKeyId"],
            access_key_secret=cred_json["AccessKeySecret"],
            security_token=cred_json["SecurityToken"],
            type=auth_constant.STS,
        )


def get_dlc_client(region_id: str = None) -> DLCJobClient:
    region_id = region_id or REGION_ID_FROM_ENV
    vpc_endpoint = f"pai-dlc-vpc.{region_id}.aliyuncs.com"
    internet_endpoint = f"pai-dlc.{region_id}.aliyuncs.com"

    endpoint = vpc_endpoint if is_reachable(vpc_endpoint) else internet_endpoint

    client = DLCJobClient(
        config=open_api_models.Config(
            credential=CustomCredentialClient(),
            endpoint=endpoint,
        )
    )
    return client


def get_langstudio_client(region_id: str = None) -> LangStudioClient:
    region_id = region_id or REGION_ID_FROM_ENV

    vpc_endpoint = f"pailangstudio-vpc.{region_id}.aliyuncs.com"
    internet_endpoint = f"pailangstudio.{region_id}.aliyuncs.com"

    endpoint = vpc_endpoint if is_reachable(vpc_endpoint) else internet_endpoint
    client = LangStudioClient(
        config=open_api_models.Config(
            credential=CustomCredentialClient(),
            endpoint=endpoint,
        )
    )
    return client


def get_workspace_client(region_id: str = None) -> AIWorkSpaceClient:
    region_id = region_id or REGION_ID_FROM_ENV
    vpc_endpoint = f"aiworkspace-vpc.{region_id}.aliyuncs.com"
    internet_endpoint = f"aiworkspace.{region_id}.aliyuncs.com"

    endpoint = vpc_endpoint if is_reachable(vpc_endpoint) else internet_endpoint

    client = AIWorkSpaceClient(
        config=open_api_models.Config(
            credential=CustomCredentialClient(),
            endpoint=endpoint,
        )
    )
    return client
