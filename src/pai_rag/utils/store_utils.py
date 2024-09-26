import json
import os


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
