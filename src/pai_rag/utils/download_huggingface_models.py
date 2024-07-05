from abc import ABC, abstractmethod
from pai_rag.utils.constants import DEFAULT_MODEL_DIR, DEFAULT_EASYOCR_MODEL_DIR
from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile
import os


class DownloadModels(ABC):
    @abstractmethod
    def load_models_from_oss(self):
        """从oss上下载模型"""

        pass


class DownloadHuggingFaceModels(DownloadModels):
    def __init__(self):
        self.download_directory_path = Path(DEFAULT_MODEL_DIR).parent
        if not os.path.exists(self.download_directory_path):
            os.makedirs(self.download_directory_path)

    def load_models_from_oss(self):
        if not os.path.exists(DEFAULT_MODEL_DIR):
            oss_models_url = "https://pai-rag.oss-cn-hangzhou.aliyuncs.com/huggingface/sentence_transformers.zip"
            response = requests.get(oss_models_url)
            response.raise_for_status()

            with BytesIO(response.content) as zip_buffer:
                with ZipFile(zip_buffer) as zip_file:
                    zip_file.extractall(self.download_directory_path)
        else:
            print(f"transformer models already exist at {DEFAULT_MODEL_DIR}.")


class DownloadEasyOCRModels(DownloadModels):
    def __init__(self):
        self.download_directory_path = Path(DEFAULT_EASYOCR_MODEL_DIR).parent
        if not os.path.exists(self.download_directory_path):
            os.makedirs(self.download_directory_path)

    def load_models_from_oss(self):
        if not os.path.exists(DEFAULT_EASYOCR_MODEL_DIR):
            oss_models_url = (
                "https://pai-rag.oss-cn-hangzhou.aliyuncs.com/huggingface/easyocr.zip"
            )
            response = requests.get(oss_models_url)
            response.raise_for_status()

            with BytesIO(response.content) as zip_buffer:
                with ZipFile(zip_buffer) as zip_file:
                    zip_file.extractall(self.download_directory_path)
        else:
            print(f"EasyOcr models already exist at {DEFAULT_EASYOCR_MODEL_DIR}.")


def load_easyocr_models():
    download_easyocr_models = DownloadEasyOCRModels()
    download_easyocr_models.load_models_from_oss()
