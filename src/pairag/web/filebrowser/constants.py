from pairag.utils.constants import try_get_int_env

DEFAULT_FILE_BROWSER_PORT = try_get_int_env("DEFAULT_FILE_BROWSER_PORT", 8012)

FILEBROWSER_PREFIX = "/filebrowser/api/resources"
FILEBROWSER_PREFIX_LEN = len(FILEBROWSER_PREFIX)
