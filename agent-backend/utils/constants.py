import os


def try_get_int_env(key, default_value=None):
    """
    Retrieves an integer from an environment variable.
    """

    value_str = os.getenv(key)
    if value_str is None:
        if default_value is not None:
            return default_value
        else:
            return None
    try:
        return int(value_str)
    except ValueError:
        return None


BACKEND_PORT = try_get_int_env("BACKEND_PORT", 8029)
