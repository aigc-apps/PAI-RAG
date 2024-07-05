from pai_rag.utils.download_huggingface_models import (
    DownloadHuggingFaceModels,
    DownloadEasyOCRModels,
)
from pai_rag.utils.constants import DEFAULT_MODEL_DIR, DEFAULT_EASYOCR_MODEL_DIR
import os


def test_download_huggingfacemodels():
    DownloadHuggingFaceModels().load_models_from_oss()
    DownloadEasyOCRModels().load_models_from_oss()

    assert os.path.exists(DEFAULT_MODEL_DIR)
    assert os.path.exists(DEFAULT_EASYOCR_MODEL_DIR)
