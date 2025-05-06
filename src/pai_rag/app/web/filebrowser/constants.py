from pai_rag.utils.constants import try_get_int_env

DEFAULT_FILE_BROWER_PORT = try_get_int_env("DEFAULT_FILE_BROWER_PORT", 8012)

FILEBROWER_PREFIX = "/filebrowser/api/resources"
FILEBROWER_PREFIX_LEN = len(FILEBROWER_PREFIX)
