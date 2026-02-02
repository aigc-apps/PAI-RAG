"""Backend i18n (internationalization) module for PAI-RAG.

This module provides a singleton I18n class for managing translations in the backend API layer.
It loads translation resources from JSON files and provides a `t()` method for retrieving translated messages.
The locale is determined from the request's Accept-Language header (defaults to 'zh', supports 'zh' or 'en').

Usage:
    from common.i18n import i18n

    # Simple translation (locale auto-detected from request header)
    message = i18n.t("api.success.operation")

    # Translation with parameters
    message = i18n.t("api.error.not_found", resource="Knowledge Base", id="kb123")
"""

import json
import os
from typing import Dict, Any, Optional
from loguru import logger
from contextvars import ContextVar

# Context variable to store the current request's locale
_locale_context: ContextVar[Optional[str]] = ContextVar('locale', default=None)


class I18n:
    """
    Singleton class for managing backend internationalization.

    Loads translations from resources/i18n/{locale}.json and provides
    a simple key-based translation mechanism with parameter interpolation.
    The locale is determined from the request's Accept-Language header.
    """

    _instance = None
    _translations_cache: Dict[str, Dict[str, Any]] = {}  # Cache for all loaded locales
    _default_locale = "zh"  # Default to Chinese
    _supported_locales = {"zh", "en"}  # Supported locales

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(I18n, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the i18n system by preloading all translation files."""
        # Preload all supported locales
        for locale in self._supported_locales:
            self._load_translations(locale)

    def _load_translations(self, locale: str):
        """
        Load translations from JSON file and cache them.

        Args:
            locale: The locale code (e.g., 'zh', 'en')
        """
        # Skip if already loaded
        if locale in self._translations_cache:
            return

        # Get the project root directory (3 levels up from backend/common/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # Construct path to translation file
        translation_file = os.path.join(project_root, "resources", "i18n", f"{locale}.json")

        try:
            if os.path.exists(translation_file):
                with open(translation_file, "r", encoding="utf-8") as f:
                    self._translations_cache[locale] = json.load(f)
                logger.info(f"Loaded translations for locale: {locale}")
            else:
                logger.warning(f"Translation file not found: {translation_file}")
                self._translations_cache[locale] = {}
        except Exception as e:
            logger.error(f"Failed to load translations for locale {locale}: {e}")
            self._translations_cache[locale] = {}

    def _get_locale(self) -> str:
        """
        Get the current locale from context or return default.

        Returns:
            The locale code (e.g., 'zh', 'en')
        """
        locale = _locale_context.get()
        if locale and locale in self._supported_locales:
            return locale
        return self._default_locale

    def t(self, key: str, **kwargs) -> str:
        """
        Translate a message key to localized text based on the current request's locale.
        The locale is determined from the Accept-Language header (via context variable).

        Args:
            key: The translation key (dot-separated path, e.g., "api.success.create")
            **kwargs: Parameters for string interpolation (e.g., resource="KB", id="123")

        Returns:
            The translated message with parameters interpolated, or the key itself if not found.

        Examples:
            >>> i18n.t("api.success.create")  # Returns "创建成功" for zh, "Created successfully" for en

            >>> i18n.t("api.error.not_found", resource="Knowledge Base", id="kb123")
            # Returns localized message based on current request's Accept-Language header
        """
        # Get the current locale from context
        locale = self._get_locale()
        translations = self._translations_cache.get(locale, {})

        # Navigate through nested dictionary using dot-separated key
        keys = key.split(".")
        value = translations

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Key not found, return the key itself as fallback
                logger.warning(f"Translation key not found: {key}")
                return key

        # If value is not a string, return the key
        if not isinstance(value, str):
            logger.warning(f"Translation value is not a string for key: {key}")
            return key

        # Perform parameter interpolation
        if kwargs:
            try:
                # Support both {param} and ${param} style placeholders
                result = value
                for param_key, param_value in kwargs.items():
                    # Replace {key} style
                    result = result.replace(f"{{{param_key}}}", str(param_value))
                    # Replace ${key} style
                    result = result.replace(f"${{{param_key}}}", str(param_value))
                return result
            except Exception as e:
                logger.error(f"Failed to interpolate parameters for key {key}: {e}")
                return value

        return value

    def set_locale(self, locale: str):
        """
        Set the default locale for the application.
        Note: For request-specific locale, use set_request_locale() instead.

        Args:
            locale: The locale code (e.g., 'zh', 'en')
        """
        if locale in self._supported_locales:
            self._default_locale = locale
            # Ensure the locale is loaded
            self._load_translations(locale)
        else:
            logger.warning(f"Unsupported locale: {locale}. Supported locales: {self._supported_locales}")


def set_request_locale(locale: str):
    """
    Set the locale for the current request context.
    This is typically called by middleware based on the Accept-Language header.

    Args:
        locale: The locale code (e.g., 'zh', 'en')
    """
    if locale in i18n._supported_locales:
        _locale_context.set(locale)
    else:
        # Default to 'zh' if unsupported locale
        _locale_context.set('zh')


def get_request_locale() -> str:
    """
    Get the locale for the current request context.

    Returns:
        The locale code (e.g., 'zh', 'en')
    """
    return _locale_context.get() or 'zh'


# Singleton instance
i18n = I18n()
