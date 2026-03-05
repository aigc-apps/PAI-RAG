"""HTTP Request Tool - Make HTTP requests for skill use.

Allows skills to interact with external APIs and web services.
"""

import json
import asyncio
from typing import Optional
from loguru import logger
from llama_index.core.tools import FunctionTool

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Maximum response size
MAX_RESPONSE_SIZE = 100000
# Default timeout
DEFAULT_TIMEOUT = 30


async def _http_request(
    url: str,
    method: str = "GET",
    headers: Optional[str] = None,
    body: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Make an HTTP request and return the response.

    Args:
        url: The URL to send the request to.
        method: HTTP method (GET, POST, PUT, DELETE, PATCH). Default: GET.
        headers: Optional JSON string of request headers.
            Example: '{"Authorization": "Bearer xxx", "Content-Type": "application/json"}'
        body: Optional request body (string). For JSON, provide a JSON string.
        timeout: Request timeout in seconds (default: 30).

    Returns:
        A string containing the response status, headers, and body.
    """
    if not HAS_AIOHTTP:
        return "Error: aiohttp library is not installed. Run: pip install aiohttp"

    if not url or not url.strip():
        return "Error: URL is required."

    method = method.upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
        return f"Error: Unsupported HTTP method: {method}"

    # Parse headers
    parsed_headers = {}
    if headers:
        try:
            parsed_headers = json.loads(headers)
        except json.JSONDecodeError:
            return "Error: Invalid JSON in headers parameter."

    logger.info(f"HTTP {method} {url}")

    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.request(
                method=method,
                url=url,
                headers=parsed_headers,
                data=body.encode("utf-8") if body else None,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                status = response.status
                resp_headers = dict(response.headers)

                # Read response body
                try:
                    resp_body = await response.text()
                except Exception:
                    resp_body = "(binary or unreadable response)"

                if len(resp_body) > MAX_RESPONSE_SIZE:
                    resp_body = resp_body[:MAX_RESPONSE_SIZE] + "\n...(response truncated)"

                result_parts = [
                    f"Status: {status}",
                    f"Headers: {json.dumps(dict(list(resp_headers.items())[:20]), ensure_ascii=False)}",
                    f"Body:\n{resp_body}",
                ]

                return "\n\n".join(result_parts)

    except asyncio.TimeoutError:
        return f"Error: Request timed out after {timeout} seconds."
    except aiohttp.ClientError as e:
        return f"Error: HTTP request failed: {e}"
    except Exception as e:
        return f"Error: Unexpected error: {e}"


def create_http_request_tool() -> FunctionTool:
    """Create the http_request FunctionTool."""
    return FunctionTool.from_defaults(
        async_fn=_http_request,
        name="http_request",
        description=(
            "Make an HTTP request to a URL and return the response. "
            "Parameters: url (str, required) - the target URL; "
            "method (str, optional) - HTTP method (GET/POST/PUT/DELETE/PATCH, default: GET); "
            "headers (str, optional) - JSON string of request headers; "
            "body (str, optional) - request body string; "
            "timeout (int, optional) - timeout in seconds (default: 30). "
            "Returns status code, response headers, and response body."
        ),
        return_direct=False,
    )
