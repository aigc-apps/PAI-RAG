from pathlib import Path
import os

_BASE_DIR = Path(__file__).parent.parent
_ROOT_BASE_DIR = Path(__file__).parent.parent.parent.parent

DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")
DEFAULT_APPLICATION_EXAMPLE_DATA_FILE = os.path.join(
    _ROOT_BASE_DIR, "example_data/pai_document.pdf"
)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_RAG_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/"
