import base64
import requests
from binascii import Error as BinasciiError
from io import BytesIO
from pathlib import Path
from typing import (
    Optional,
    Union,
)


def resolve_binary(
    raw_bytes: Optional[bytes] = None,
    path: Optional[Union[str, Path]] = None,
    url: Optional[str] = None,
    as_base64: bool = False,
) -> BytesIO:
    """
    Resolve binary data from various sources into a BytesIO object.

    Args:
        raw_bytes: Raw bytes data
        path: File path to read bytes from
        url: URL to fetch bytes from
        as_base64: Whether to base64 encode the output bytes

    Returns:
        BytesIO object containing the binary data

    Raises:
        ValueError: If no valid source is provided

    """
    if raw_bytes is not None:
        # check if raw_bytes is base64 encoded
        try:
            decoded_bytes = base64.b64decode(raw_bytes)
        except Exception:
            decoded_bytes = raw_bytes

        try:
            # Check if raw_bytes is already base64 encoded.
            # b64decode() can succeed on random binary data, so we
            # pass verify=True to make sure it's not a false positive
            decoded_bytes = base64.b64decode(raw_bytes, validate=True)
        except BinasciiError:
            # b64decode failed, leave as is
            decoded_bytes = raw_bytes

        if as_base64:
            return BytesIO(base64.b64encode(decoded_bytes))
        return BytesIO(decoded_bytes)

    elif path is not None:
        path = Path(path) if isinstance(path, str) else path
        data = path.read_bytes()
        if as_base64:
            return BytesIO(base64.b64encode(data))
        return BytesIO(data)

    elif url is not None:
        response = requests.get(url)
        response.raise_for_status()
        if as_base64:
            return BytesIO(base64.b64encode(response.content))
        return BytesIO(response.content)

    raise ValueError("No valid source provided to resolve binary data!")
