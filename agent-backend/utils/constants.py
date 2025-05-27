import os
from dotenv import load_dotenv


def get_int_env(key, default_value=None):
    """
    Retrieves an integer from an environment variable.
    """

    value_str = os.getenv(key)
    if value_str:
        try:
            return int(value_str)
        except ValueError:
            return default_value
    return default_value


load_dotenv(".env")
BACKEND_PORT = get_int_env("BACKEND_PORT", 8039)
