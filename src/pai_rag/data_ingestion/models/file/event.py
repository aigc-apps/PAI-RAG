from enum import Enum


class NodeOperationType(str, Enum):
    ADD = "add"
    DELETE = "delete"


class FileChangeType(str, Enum):
    ADD = "add_file"
    MODIFY = "modify_file"
    DELETE = "delete_file"
