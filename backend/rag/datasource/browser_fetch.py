"""Optional Playwright fallback for anti-bot challenge pages.

When the lightweight urllib3 fetch (``http_util.http_get``) keeps getting an
anti-bot / WAF challenge (e.g. Aliyun x5sec), a real browser can run the
challenge JavaScript and complete the cookie handshake, then fetch the true
body with that cookie. This is opt-in
(``PAIRAG_DATASOURCE_BROWSER_FALLBACK=true``) and requires Playwright plus a
Chromium install::

    pip install playwright && playwright install chromium

Limits: it solves the JS/cookie class of challenge, not interactive
(slider/click) captchas. SSRF: the host is validated as public up front, but —
unlike ``http_get`` — the browser does its own DNS resolution, so the
connection is not IP-pinned (the DNS-rebinding window reopens). Keep the
fallback for trusted public doc hosts.
"""

import concurrent.futures

from loguru import logger

from rag.datasource import http_util
from rag.datasource.url_guard import validate_public_url
from utils.constants import try_get_int_env

# Browser navigation is slower than a raw GET; give it its own (larger) budget.
BROWSER_TIMEOUT = try_get_int_env("PAIRAG_DATASOURCE_BROWSER_TIMEOUT", 45)


def _run_sync(url: str, timeout: int) -> str:
    """Drive Chromium synchronously and return the post-challenge body text.

    Runs in a dedicated worker thread (see ``browser_get``) so the Playwright
    sync API never collides with a running asyncio loop.
    """
    from playwright.sync_api import sync_playwright  # heavy + optional

    ms = max(1, timeout) * 1000
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                user_agent=http_util.UA,
                locale="zh-CN",
                extra_http_headers={"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=ms)
            # Let the challenge JS run, redirect, and set its cookie.
            try:
                page.wait_for_load_state("networkidle", timeout=ms)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"browser_get: networkidle wait skipped for '{url}': {e}")
            # With the challenge cookie now in the jar, fetch the real body
            # (avoids DOM-wrapping a markdown/text response).
            try:
                resp = context.request.get(url, timeout=ms)
                return resp.text()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"browser_get: request-after-solve failed for '{url}': {e}; using page content.")
                return page.content()
        finally:
            browser.close()


def browser_get(url: str, timeout: int = BROWSER_TIMEOUT) -> str:
    """Fetch ``url`` with a headless browser, solving JS/cookie anti-bot challenges.

    Raises ``http_util.ChallengeDetected`` if the page is still a challenge after
    the browser ran (e.g. an interactive captcha it cannot solve).
    """
    validate_public_url(url)  # SSRF: reject non-public hosts before launching
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        body = ex.submit(_run_sync, url, timeout).result()
    if http_util._looks_like_challenge(body):
        raise http_util.ChallengeDetected(
            f"Browser fallback still hit an anti-bot challenge for '{url}'."
        )
    return body
