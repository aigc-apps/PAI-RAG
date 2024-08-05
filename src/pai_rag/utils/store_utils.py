import hashlib
import json
import os


def get_store_persist_directory_name(storage_config, ndims):
    """Generate directory name to persist docs/indexes based on user store config.

    Used for multiple store config scenario where user may want to connect to FAISS/Hologres/Adb etc.
    In this scenario, we want to keep a separate folder for each config to keep the store clean.
    Here we will compute the SHA256 hash from config to get directory name.
    """
    raw_text = "sample_store_key"
    vector_store_type = storage_config["vector_store"]["type"].lower()
    if vector_store_type == "chroma":
        raw_text = json.dumps(storage_config["vector_store"], sort_keys=True)
    elif vector_store_type == "faiss":
        raw_text = {"type": "faiss"}
    elif vector_store_type == "hologres":
        keywords = ["host", "port", "database", "table_name"]
        json_data = {k: storage_config["vector_store"][k] for k in keywords}
        raw_text = json.dumps(json_data)
    elif vector_store_type == "analyticdb":
        keywords = ["region_id", "instance_id", "namespace", "collection"]
        json_data = {k: storage_config["vector_store"][k] for k in keywords}
        raw_text = json.dumps(json_data)
    elif vector_store_type == "elasticsearch":
        keywords = ["es_url", "es_index"]
        json_data = {k: storage_config["vector_store"][k] for k in keywords}
        raw_text = json.dumps(json_data)
    elif vector_store_type == "milvus":
        keywords = ["host", "port", "database", "collection_name"]
        json_data = {k: storage_config["vector_store"][k] for k in keywords}
        raw_text = json.dumps(json_data)
    elif vector_store_type == "opensearch":
        keywords = ["endpoint", "instance_id", "table_name"]
        json_data = {k: storage_config["vector_store"][k] for k in keywords}
        raw_text = json.dumps(json_data)
    elif vector_store_type == "postgresql":
        keywords = ["host", "port", "database", "table_name"]
        json_data = {k: storage_config["vector_store"][k] for k in keywords}
        raw_text = json.dumps(json_data)
    else:
        raise ValueError(f"Unknown vector_store type '{vector_store_type}'.")

    encoded_raw_text = f"{raw_text}_{ndims}".encode()
    hash = hashlib.sha256(encoded_raw_text).hexdigest()
    return hash


def save_index_state(state, file_path):
    with open(file_path, "w") as file:
        json.dump(state, file)


def read_index_state(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception:
        return None  # 如果文件不存在/json不合法，则返回None


def read_chat_store_state(persist_dir, file_path):
    chat_store_path = os.path.join(persist_dir, file_path)
    try:
        with open(chat_store_path, "r") as file:
            return json.load(file)
    except Exception:
        return None  # 如果文件不存在/json不合法，则返回None
