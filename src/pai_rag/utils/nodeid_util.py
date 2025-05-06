import hashlib


def compute_node_id(i: int, file_name: str):
    encoded_raw_text = f"""<<{i}>>{file_name}""".encode()
    hash = hashlib.sha256(encoded_raw_text).hexdigest()
    return hash
