import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.tools.builtin.datetime_tool import make_current_datetime_tool
from agent.tools.builtin.web_fetch import make_web_fetch_tool
from agent.tools.builtin.web_search import make_web_search_tool
from agent.tools.defaults import build_default_registry


def test_current_datetime_tool_returns_a_time_string():
    t = make_current_datetime_tool()
    assert t.name == "current_datetime"
    out = asyncio.run(t.fn())
    assert isinstance(out, str) and len(out) >= 8


class _FakeResp:
    def __init__(self, text):
        self.text = text
    def raise_for_status(self):
        return None


class _FakeClient:
    def __init__(self, resp, boom=False):
        self._resp = resp
        self._boom = boom
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def get(self, url):
        if self._boom:
            raise RuntimeError("network down")
        return self._resp


def test_web_fetch_extracts_text_and_truncates():
    html = "<html><head><style>x{}</style></head><body><h1>Hi</h1><p>World &amp; more</p><script>bad()</script></body></html>"
    t = make_web_fetch_tool(client_factory=lambda: _FakeClient(_FakeResp(html)), limit=50)
    out = asyncio.run(t.fn(url="http://x"))
    assert "Hi" in out and "World" in out
    assert "bad()" not in out and "<" not in out
    assert len(out) <= 50


def test_web_fetch_returns_error_string_on_failure():
    t = make_web_fetch_tool(client_factory=lambda: _FakeClient(None, boom=True))
    out = asyncio.run(t.fn(url="http://x"))
    assert out.startswith("web_fetch failed:")


class _FakeProvider:
    async def search(self, query, num_results):
        return [{"title": "T1", "url": "http://1", "snippet": "S1"},
                {"title": "T2", "url": "http://2", "snippet": "S2"}][:num_results]


def test_web_search_formats_results():
    t = make_web_search_tool(_FakeProvider())
    out = asyncio.run(t.fn(query="hello", num_results=2))
    assert "T1" in out and "http://2" in out and "S2" in out


def test_web_search_returns_error_string_on_provider_failure():
    class _Boom:
        async def search(self, q, n):
            raise RuntimeError("boom")
    t = make_web_search_tool(_Boom())
    out = asyncio.run(t.fn(query="x"))
    assert out.startswith("web_search failed:")


class _Settings:
    search_provider = "none"


def test_default_registry_omits_search_when_unconfigured():
    reg = build_default_registry(_Settings())
    assert set(reg.names()) == {"current_datetime", "web_fetch"}


def test_default_registry_includes_injected_search_provider():
    reg = build_default_registry(_Settings(), search_provider=_FakeProvider())
    assert "web_search" in reg.names()
