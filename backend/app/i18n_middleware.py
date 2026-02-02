"""
Middleware for handling i18n locale detection from Accept-Language header.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from common.i18n import set_request_locale
from loguru import logger


class I18nMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts the Accept-Language header from requests
    and sets the appropriate locale for the i18n system.

    Supported locales: 'zh' (Chinese, default), 'en' (English)
    """

    async def dispatch(self, request: Request, call_next):
        # Get Accept-Language header (default to 'zh')
        accept_language = request.headers.get("Accept-Language", "zh").lower()

        # Parse the Accept-Language header (take the first language code)
        # Format can be: 'zh', 'en', 'zh-CN', 'en-US', 'zh,en;q=0.9', etc.
        locale = self._parse_accept_language(accept_language)

        # Set the locale for this request context
        set_request_locale(locale)

        logger.debug(f"Request locale set to: {locale}")

        # Process the request
        response = await call_next(request)

        return response

    def _parse_accept_language(self, accept_language: str) -> str:
        """
        Parse the Accept-Language header to extract the primary locale.

        Args:
            accept_language: The Accept-Language header value

        Returns:
            The locale code ('zh' or 'en')
        """
        if not accept_language:
            return "zh"

        # Split by comma to get multiple languages (if present)
        languages = accept_language.split(",")

        if not languages:
            return "zh"

        # Take the first language (highest priority)
        primary_lang = languages[0].strip()

        # Remove quality factor (e.g., 'en;q=0.9' -> 'en')
        if ";" in primary_lang:
            primary_lang = primary_lang.split(";")[0].strip()

        # Extract the language code (e.g., 'zh-CN' -> 'zh', 'en-US' -> 'en')
        lang_code = primary_lang.split("-")[0].lower()

        # Map to supported locales
        if lang_code == "en":
            return "en"
        elif lang_code == "zh":
            return "zh"
        else:
            # Default to Chinese for unsupported languages
            return "zh"
