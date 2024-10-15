import json
import logging

logger = logging.getLogger(__name__)


def load_qca_dataset_json(qca_dataset_path):
    """Load json."""
    with open(qca_dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_qca_dataset_json(qas, qca_dataset_path: str) -> None:
    """Save json."""
    with open(qca_dataset_path, "w", encoding="utf-8") as f:
        json.dump(qas.dict(), f, indent=4, ensure_ascii=False)
    print(f"Saved dataset to {qca_dataset_path}.")
