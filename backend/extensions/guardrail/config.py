import os
from utils.constants import try_get_int_env

CHECK_INPUT_ALGO = os.environ.get("INPUT_CHECK_ALGO", "query_security_check")
CHECK_OUTPUT_AGLO = os.environ.get("OUTPUT_CHECK_ALGO", "response_security_check")
CHECK_OUTPUT_CHUNK_SIZE = try_get_int_env("OUTPUT_CHECK_CHUNK_SIZE", 200)
CHECK_OUTPUT_CHUNK_OVERLAP = try_get_int_env("OUTPUT_CHECK_CHUNK_OVERLAP", 20)
