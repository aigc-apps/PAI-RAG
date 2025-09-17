import os
from cryptography.fernet import Fernet


# 默认的key，不随机生成担心服务重启key丢失，建议用户指定自己的key
DEFAULT_ENCRYPTION_KEY = "VaR7rHz4ZhULu1injtEMVCvZICA241Mny42YCUOs7ag="

# 从环境变量读取密钥
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", DEFAULT_ENCRYPTION_KEY).encode()
cipher = Fernet(ENCRYPTION_KEY)


def encrypt_key(key: str) -> str:
    if not key:
        return key

    return cipher.encrypt(key.encode()).decode()


def decrypt_key(key: str) -> str:
    if not key:
        return key

    return cipher.decrypt(key.encode()).decode()
