import os
import pytest
import requests
from pai_rag.file.store.oss_store import PaiOssStore
from pai_rag.file.store.pai_image_store import PaiImageStore
from pai_rag.file.readers.pai.file_readers.pai_image_reader import PaiImageReader

if not os.environ.get("OSS_ACCESS_KEY_ID") or not os.environ.get(
    "OSS_ACCESS_KEY_SECRET"
):
    pytest.skip(
        reason="OSS_ACCESS_KEY_ID or OSS_ACCESS_KEY_SECRET not set",
        allow_module_level=True,
    )


@pytest.fixture
def image_store():
    oss_store = PaiOssStore(
        bucket_name="feiyue-test", endpoint="oss-cn-hangzhou.aliyuncs.com"
    )
    return PaiImageStore(oss_store=oss_store)


def test_image_reader(image_store):
    image_reader = PaiImageReader(image_store=image_store)
    test_image_path = "tests/testdata/data/image_data/11.jpg"
    image_doc = image_reader.load_data(file_path=test_image_path)[0]
    image_url = image_doc.metadata.get("image_url")
    assert image_url is not None, "image url should not be None."

    image_response = requests.get(image_url)
    assert image_response.status_code == 200, "image url should be valid."
