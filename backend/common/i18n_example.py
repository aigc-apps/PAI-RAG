"""
Example usage of the i18n system with Accept-Language header support.

This demonstrates how the i18n system automatically detects the locale
from the request's Accept-Language header.
"""

from common.i18n import i18n, set_request_locale, get_request_locale


def example_usage():
    """
    Example showing how i18n works with different locales.
    In production, the locale is automatically set by the I18nMiddleware
    based on the Accept-Language header.
    """

    print("=== i18n Example Usage ===\n")

    # Example 1: Chinese locale (default)
    print("1. Chinese locale (zh):")
    set_request_locale("zh")
    print(f"   Current locale: {get_request_locale()}")
    print(f"   Success message: {i18n.t('api.success.create')}")
    print(f"   Error message: {i18n.t('api.error.not_found', resource='知识库', id='kb123')}")
    print()

    # Example 2: English locale
    print("2. English locale (en):")
    set_request_locale("en")
    print(f"   Current locale: {get_request_locale()}")
    print(f"   Success message: {i18n.t('api.success.create')}")
    print(f"   Error message: {i18n.t('api.error.not_found', resource='Knowledge Base', id='kb123')}")
    print()

    # Example 3: With different messages
    print("3. Various messages in English:")
    set_request_locale("en")
    print(f"   Upload: {i18n.t('api.attachment.upload_success', filename='test.pdf')}")
    print(f"   FAQ: {i18n.t('api.faq.create_success')}")
    print(f"   LLM: {i18n.t('api.llm.list_success')}")
    print()

    # Example 4: Same messages in Chinese
    print("4. Same messages in Chinese:")
    set_request_locale("zh")
    print(f"   Upload: {i18n.t('api.attachment.upload_success', filename='test.pdf')}")
    print(f"   FAQ: {i18n.t('api.faq.create_success')}")
    print(f"   LLM: {i18n.t('api.llm.list_success')}")
    print()


if __name__ == "__main__":
    example_usage()
