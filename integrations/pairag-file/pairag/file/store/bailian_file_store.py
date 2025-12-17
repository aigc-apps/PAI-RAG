from io import BytesIO
import os
import aiohttp
import requests
from pairag.file.store.base import BaseFileStore, FileUploadResult
from loguru import logger
from typing import Optional, BinaryIO
import traceback

DEFAULT_FILE_TYPE = "document"

class BailianFileStore(BaseFileStore):
    def __init__(self):
        super().__init__()
        self.endpoint = os.environ.get("BAILIAN_CONSOLE_ENDPOINT", "").rstrip("/")
        assert self.endpoint, "BAILIAN_CONSOLE_ENDPOINT is not set."
        logger.info(f"BailianFileStore initialized with endpoint: {self.endpoint}")

    def get_url(self, file_path: str, tenant_id: str) -> Optional[str]:
        try:
            logger.info(f"Getting url for file {file_path} from BailianFileStore.")
            response = requests.get(f"{self.endpoint}/infra/v1/files/download-url", params={"object_name": file_path}, headers={"X-TENANT-ID": tenant_id})
            if response.status_code != 200:
                logger.error(f"Failed to get_url for file {file_path}. status: {response.status_code}, response: {response.text}")
                raise Exception(f"Failed to get_url for file {file_path}. status: {response.status_code}, response: {response.text}")
            
            url = response.json()["data"]["url"]
            logger.info(f"Got url for file {file_path} from BailianFileStore successfully.")
            return url
        except Exception as e:
            logger.error(f"Failed to get url for file {file_path}. error: {e}")
            raise
    
    def write(self, file: BinaryIO, file_name: str, file_path: str, tenant_id: str) -> FileUploadResult:
        try:
            logger.info(f"Writing file {file_name} to BailianFileStore.")
            files = {"file": (file_name, file.read())}

            response = requests.post(
                f"{self.endpoint}/infra/v1/files/upload",
                files=files,
                data={"file_type": DEFAULT_FILE_TYPE},
                headers={"X-TENANT-ID": tenant_id},
            )
            if response.status_code != 200:
                logger.error(f"Failed to write file {file_path}. status: {response.status_code}, response: {response.text}")
                raise Exception(f"Failed to write file {file_path}. status: {response.status_code}, response: {response.text}")
            response_data = response.json()
            return FileUploadResult(
                file_name=response_data["data"]["file_name"],
                file_path=response_data["data"]["object_name"],
            )
        except Exception as e:
            logger.error(f"Failed to write file {file_path}. error: {e}")
            raise
    
    def read(self, file_path: str, tenant_id: str) -> Optional[BinaryIO]:
        try:
            logger.info(f"Reading file {file_path} from BailianFileStore.")
            response = requests.get(f"{self.endpoint}/infra/v1/files/download", params={"object_name": file_path}, headers={"X-TENANT-ID": tenant_id})
            if response.status_code != 200:
                logger.error(f"Failed to read file {file_path}. status: {response.status_code}, response: {response.text}")
                raise Exception(f"Failed to read file {file_path}. status: {response.status_code}, response: {response.text}")
            logger.info(f"Read file {file_path} from BailianFileStore successfully.")
            return BytesIO(response.content)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}. error: {e}")
            raise

    async def get_url_async(self, file_path: str, tenant_id: str) -> Optional[str]:
        try:
            logger.info(f"Getting url for file {file_path} from BailianFileStore asynchronously.")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.endpoint}/infra/v1/files/download-url", params={"object_name": file_path}, headers={"X-TENANT-ID": tenant_id}) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return response_data["data"]["url"]
                    else:
                        raise Exception(f"Failed to get url for file {file_path}. status: {response.status}, response: {await response.text()}")
        except Exception as e:
            logger.error(f"Failed to get url for file {file_path}. error: {e}")
            return None

    async def write_async(self, file: BinaryIO, file_name: str, file_path: str, tenant_id: str) -> str:
        try:
            logger.info(f"Writing file {file_name} to BailianFileStore asynchronously.")
            form_data = aiohttp.FormData()
            form_data.add_field("file", file.read(), filename=file_name)
            form_data.add_field("file_type", DEFAULT_FILE_TYPE)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.endpoint}/infra/v1/files/upload", 
                    data=form_data,
                    headers={"X-TENANT-ID": tenant_id}) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return FileUploadResult(
                            file_name=response_data["data"]["file_name"],
                            file_path=response_data["data"]["object_name"],
                        )
                    else:
                        raise Exception(f"Failed to write file {file_path}. status: {response.status}, response: {await response.text()}")
        except Exception as e:
            logger.error(f"Failed to write file {file_path}. error: {traceback.format_exc()}")
            raise


    async def read_async(self, file_path: str, tenant_id: str) -> Optional[BinaryIO]:
        try:
            logger.info(f"Reading file {file_path} from BailianFileStore asynchronously.")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.endpoint}/infra/v1/files/download", params={"object_name": file_path}, headers={"X-TENANT-ID": tenant_id}) as response:
                    if response.status == 200:
                        logger.info(f"Read file {file_path} from BailianFileStore asynchronously successfully.")
                        return BytesIO(await response.read())
                    else:
                        logger.error(f"Failed to read file {file_path}. status: {response.status}, response: {await response.text()}")
                        raise Exception(f"Failed to read file {file_path}. status: {response.status}, response: {await response.text()}")
        except Exception as e:
            logger.error(f"Failed to read file {file_path}. error: {traceback.format_exc()}")
            raise

    async def exists_async(self, file_path: str, tenant_id: str) -> bool:
        raise NotImplementedError("BailianFileStore does not support exists_async")

if __name__ == "__main__":
    import asyncio
    async def test_async():
        file_store = BailianFileStore()
        file_path = "tests/testdata/pai_document.md"
        tenant_id = "7202"
        with open(file_path, "rb") as f:
            result = await file_store.write_async(f, file_name=os.path.basename(file_path), file_path=file_path, tenant_id=tenant_id)
            print(result)
            url = await file_store.get_url_async(file_path, tenant_id)
            print(url)
            file = await file_store.read_async(result.file_path, tenant_id)
            print(file.read().decode("utf-8"))
    
    def test():
        file_store = BailianFileStore()
        file_path = "tests/testdata/pai_document.md"
        tenant_id = "7202"
        with open(file_path, "rb") as f:
            result = file_store.write(f, file_name=os.path.basename(file_path), file_path=file_path, tenant_id=tenant_id)
            print(result)
            url = file_store.get_url(file_path, tenant_id)
            print(url)
            file = file_store.read(result.file_path, tenant_id)
            print(file.read().decode("utf-8"))
    test()