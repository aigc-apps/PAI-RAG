import contextlib
import importlib.util
import io
import json
import os
import urllib.error

# Load the CLI script under a unique module name. A plain `import pairag` would
# collide with the backend's real `pairag` package (src/pairag): in a full test
# run that package is already in sys.modules and would shadow this script.
_spec = importlib.util.spec_from_file_location(
    "pairag_cli_under_test", os.path.join(os.path.dirname(__file__), "pairag.py")
)
pairag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pairag)


class _Args:
    """Minimal stand-in for an argparse Namespace."""

    def __init__(self, **kw):
        self.base_url = kw.get("base_url")
        self.tenant = kw.get("tenant")
        self.token = kw.get("token")


# --------------------------------------------------------------------------- #
# Task 1: config resolution + error type
# --------------------------------------------------------------------------- #
def test_config_defaults_when_nothing_set():
    cfg = pairag.resolve_config(_Args(), env={})
    assert cfg.base_url == "http://localhost:8682"
    assert cfg.tenant is None
    assert cfg.default_kb is None
    assert cfg.token is None


def test_config_precedence_flag_over_env():
    env = {
        "PAIRAG_BASE_URL": "http://env:2",
        "PAIRAG_TENANT_ID": "tenv",
        "PAIRAG_KB": "kbenv",
        "PAIRAG_TOKEN": "tokenv",
    }
    args = _Args(base_url="http://flag:3")
    cfg = pairag.resolve_config(args, env=env)
    assert cfg.base_url == "http://flag:3"  # flag wins over env
    assert cfg.tenant == "tenv"  # env used when no flag
    assert cfg.default_kb == "kbenv"  # PAIRAG_KB → default_kb
    assert cfg.token == "tokenv"


def test_config_empty_env_value_falls_through_to_default():
    cfg = pairag.resolve_config(_Args(), env={"PAIRAG_TENANT_ID": ""})
    assert cfg.tenant is None  # empty string treated as unset


def test_pairag_error_is_exception():
    assert issubclass(pairag.PairagError, Exception)


# --------------------------------------------------------------------------- #
# Task 2: HTTP client
# --------------------------------------------------------------------------- #
def _client(opener=None):
    return pairag.Client(
        pairag.Config("http://localhost:8682", None, None, None), opener=opener
    )


def test_request_builds_url_headers_and_parses_json():
    captured = {}

    def opener(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["tenant"] = req.headers.get("X-tenant-id")  # urllib title-cases keys
        return io.BytesIO(json.dumps({"code": 200, "data": {"ok": True}}).encode())

    c = pairag.Client(pairag.Config("http://host:8682", "acme", None, None), opener=opener)
    out = c.get("/v1/config/knowledgebases", {"size": 10, "query": None})
    assert out == {"code": 200, "data": {"ok": True}}
    assert captured["url"] == "http://host:8682/v1/config/knowledgebases?size=10"
    assert captured["method"] == "GET"
    assert captured["tenant"] == "acme"


def test_post_sends_json_body():
    captured = {}

    def opener(req, timeout=None):
        captured["body"] = req.data
        captured["ctype"] = req.headers.get("Content-type")
        return io.BytesIO(json.dumps({"records": []}).encode())

    c = _client(opener=opener)
    c.post("/v1/retrieval", {"query": "hi", "knowledge_id": "k1"})
    assert json.loads(captured["body"]) == {"query": "hi", "knowledge_id": "k1"}
    assert captured["ctype"] == "application/json"


def test_connection_refused_is_friendly():
    def opener(req, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError("Connection refused"))

    c = _client(opener=opener)
    try:
        c.get("/v1/config/knowledgebases")
        assert False, "expected PairagError"
    except pairag.PairagError as e:
        assert "server running" in str(e).lower()
        assert "http://localhost:8682" in str(e)


def test_http_error_surfaces_status_and_message():
    def opener(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url,
            404,
            "Not Found",
            hdrs=None,
            fp=io.BytesIO(json.dumps({"message": "kb missing"}).encode()),
        )

    c = _client(opener=opener)
    try:
        c.get("/v1/config/knowledgebases/x")
        assert False, "expected PairagError"
    except pairag.PairagError as e:
        assert "404" in str(e)
        assert "kb missing" in str(e)


# --------------------------------------------------------------------------- #
# Task 3: data_of + resolve_kb
# --------------------------------------------------------------------------- #
def test_data_of_unwraps_config_envelope_and_passes_flat_through():
    assert pairag.data_of({"code": 200, "data": {"x": 1}}) == {"x": 1}
    assert pairag.data_of({"records": []}) == {"records": []}  # no "data" key -> as-is


def test_resolve_kb_hex_id_short_circuits_without_network():
    c = _client(opener=None)
    c._request = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not call HTTP")
    )
    kb_id = "a" * 32
    assert pairag.resolve_kb(c, kb_id) == kb_id


def test_resolve_kb_by_name_matches_case_insensitively():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "code": 200,
        "data": {"items": [{"id": "id-1", "name": "Docs"}, {"id": "id-2", "name": "Wiki"}]},
    }
    assert pairag.resolve_kb(c, "docs") == "id-1"


def test_resolve_kb_unknown_lists_available():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {"items": [{"id": "id-1", "name": "Docs"}]},
    }
    try:
        pairag.resolve_kb(c, "ghost")
        assert False, "expected PairagError"
    except pairag.PairagError as e:
        assert "ghost" in str(e)
        assert "Docs" in str(e)


def test_resolve_kb_none_errors_helpfully():
    c = _client(opener=None)
    try:
        pairag.resolve_kb(c, None)
        assert False, "expected PairagError"
    except pairag.PairagError as e:
        assert "--kb" in str(e) or "PAIRAG_KB" in str(e)


# --------------------------------------------------------------------------- #
# Task 4: kbs command
# --------------------------------------------------------------------------- #
def _kbs_client():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {
            "items": [
                {"id": "id-1", "name": "Docs", "description": "Product docs"},
                {"id": "id-2", "name": "Wiki", "description": ""},
            ]
        },
    }
    return c


def test_cmd_kbs_markdown():
    out = pairag.cmd_kbs(_kbs_client(), query=None, as_json=False)
    assert "2 knowledge base(s):" in out
    assert "- Docs (id=id-1) — Product docs" in out
    assert "- Wiki (id=id-2)" in out


def test_cmd_kbs_json():
    out = pairag.cmd_kbs(_kbs_client(), query=None, as_json=True)
    assert json.loads(out)[0]["id"] == "id-1"


def test_cmd_kbs_empty():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {"data": {"items": []}}
    assert pairag.cmd_kbs(c, query=None, as_json=False) == "No knowledge bases found."


# --------------------------------------------------------------------------- #
# Task 5: search command
# --------------------------------------------------------------------------- #
def _search_client(records):
    c = _client(opener=None)
    sent = {}

    def fake(method, path, params=None, body=None):
        sent["method"], sent["path"], sent["body"] = method, path, body
        return {"records": records}

    c._request = fake
    c.sent = sent
    return c


def test_cmd_search_posts_to_retrieval_with_resolved_kb():
    records = [
        {
            "content": "Set vector_store.type to elasticsearch and provide the endpoint url.",
            "score": 0.87,
            "title": "Vector store",
            "url": "setup/vectordb.md",
            "metadata": {
                "doc_id": "doc-9",
                "file_path": "setup/vectordb.md",
                "file_name": "vectordb.md",
            },
        }
    ]
    c = _search_client(records)
    kb_id = "f" * 32
    out = pairag.cmd_search(c, kb_target=kb_id, query="vector index config", as_json=False)
    assert c.sent["method"] == "POST"
    assert c.sent["path"] == "/v1/retrieval"
    assert c.sent["body"] == {"query": "vector index config", "knowledge_id": kb_id}
    assert '1 result(s) for "vector index config"' in out
    assert "[0.87] Vector store" in out
    assert "doc_id=doc-9" in out
    assert "elasticsearch" in out


def test_cmd_search_empty():
    c = _search_client([])
    out = pairag.cmd_search(c, kb_target="f" * 32, query="nothing", as_json=False)
    assert out == 'No results for "nothing".'


def test_cmd_search_json_returns_records():
    records = [{"content": "x", "score": 0.5, "title": "T", "metadata": {"doc_id": "d1"}}]
    c = _search_client(records)
    out = pairag.cmd_search(c, kb_target="f" * 32, query="q", as_json=True)
    assert json.loads(out) == records


# --------------------------------------------------------------------------- #
# Task 6: catalog command
# --------------------------------------------------------------------------- #
def test_cmd_catalog_lists_kb_files_brief():
    c = _client(opener=None)
    sent = {}

    def fake(method, path, params=None, body=None):
        sent["path"], sent["params"] = path, params
        return {
            "data": {
                "items": [
                    {
                        "id": "f1",
                        "file_name": "install.md",
                        "file_source": "https://example/install.md",
                        "file_metadata": {"title": "Install guide"},
                    },
                    {
                        "id": "f2",
                        "file_name": "notes.txt",
                        "file_source": None,
                        "file_metadata": {},
                    },
                ],
                "total": 2,
            }
        }

    c._request = fake
    out = pairag.cmd_catalog(c, kb_target="f" * 32, query="install", limit=20, as_json=False)
    assert sent["path"] == "/v1/config/knowledgebases/" + "f" * 32 + "/files"
    assert sent["params"] == {"query": "install", "size": 20}
    assert "2 file(s)" in out
    assert (
        "- Install guide · install.md · file_id=f1 · https://example/install.md" in out
    )
    assert "- notes.txt · file_id=f2" in out


def test_cmd_catalog_empty():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {"items": [], "total": 0}
    }
    out = pairag.cmd_catalog(c, kb_target="f" * 32, query=None, limit=20, as_json=False)
    assert out == "No files found."


# --------------------------------------------------------------------------- #
# Task 7: grep command
# --------------------------------------------------------------------------- #
def test_cmd_grep_markdown_with_counts():
    c = _client(opener=None)
    sent = {}

    def fake(method, path, params=None, body=None):
        sent["path"], sent["params"] = path, params
        return {
            "data": {
                "results": [
                    {
                        "doc_id": "d1",
                        "file_id": "f1",
                        "line": 42,
                        "match": "timeout = 600",
                        "context": "before line\ntimeout = 600\nafter line",
                        "source_url": None,
                        "title": "config.py",
                    },
                ],
                "scanned_files": 7,
                "scan_capped": False,
                "limit_reached": False,
            }
        }

    c._request = fake
    out = pairag.cmd_grep(
        c, kb_target="f" * 32, pattern="timeout", context=2, limit=20, as_json=False
    )
    assert sent["path"] == "/v1/config/knowledgebases/" + "f" * 32 + "/keyword"
    assert sent["params"] == {"pattern": "timeout", "context": 2, "limit": 20}
    assert '1 match(es) for "timeout" (scanned 7 file(s))' in out
    assert "- config.py · doc_id=d1 · line 42" in out
    # the full context block is rendered, not just the match line
    assert "    before line" in out
    assert "    timeout = 600" in out
    assert "    after line" in out


def test_cmd_grep_empty():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {
            "results": [],
            "scanned_files": 3,
            "scan_capped": False,
            "limit_reached": False,
        }
    }
    out = pairag.cmd_grep(
        c, kb_target="f" * 32, pattern="zzz", context=2, limit=20, as_json=False
    )
    assert out == 'No matches for "zzz" (scanned 3 file(s)).'


# --------------------------------------------------------------------------- #
# Task 8: read command
# --------------------------------------------------------------------------- #
def test_cmd_read_sends_doc_id_and_renders_header_plus_content():
    c = _client(opener=None)
    sent = {}

    def fake(method, path, params=None, body=None):
        sent["path"], sent["params"] = path, params
        return {
            "data": {
                "file_id": "f1",
                "file_name": "vectordb.md",
                "title": "Vector store",
                "source_url": "https://example/vectordb.md",
                "doc_id": "d1",
                "content": "# Vector store\nSet the type.",
                "content_length": 28,
                "offset": 0,
                "returned_chars": 28,
                "truncated": False,
                "next_offset": None,
                "degraded": None,
                "metadata": {},
            }
        }

    c._request = fake
    out = pairag.cmd_read(
        c, kb_target="f" * 32, ident="d1", max_chars=None, offset=0, as_json=False
    )
    assert sent["path"] == "/v1/config/knowledgebases/" + "f" * 32 + "/file-content"
    assert sent["params"] == {"doc_id": "d1", "max_chars": None, "offset": 0}
    assert "Vector store" in out
    assert "file_id=f1" in out
    assert "Set the type." in out


def test_cmd_read_truncation_note():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {
            "file_id": "f1",
            "file_name": "big.md",
            "title": "Big",
            "source_url": None,
            "doc_id": "d1",
            "content": "partial",
            "content_length": 100,
            "offset": 0,
            "returned_chars": 7,
            "truncated": True,
            "next_offset": 7,
            "degraded": None,
            "metadata": {},
        }
    }
    out = pairag.cmd_read(
        c, kb_target="f" * 32, ident="d1", max_chars=7, offset=0, as_json=False
    )
    assert "truncated" in out.lower()
    assert "--offset 7" in out


# --------------------------------------------------------------------------- #
# Task 9: argparse + dispatch + main
# --------------------------------------------------------------------------- #
def _run_main(argv, fake_client):
    """Run main() with pairag.Client replaced by a factory returning fake_client."""
    original = pairag.Client
    pairag.Client = lambda config, opener=None: fake_client
    buf = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = pairag.main(argv)
    finally:
        pairag.Client = original
    return code, buf.getvalue(), err.getvalue()


def test_main_kbs_prints_and_exits_zero():
    fake = _kbs_client()
    code, out, err = _run_main(["kbs"], fake)
    assert code == 0
    assert "knowledge base(s)" in out


def test_main_json_flag_after_subcommand():
    fake = _kbs_client()
    code, out, _ = _run_main(["kbs", "--json"], fake)
    assert code == 0
    assert json.loads(out)[0]["id"] == "id-1"


def test_main_global_kb_before_subcommand():
    fake = _search_client(
        [{"content": "x", "score": 0.1, "title": "T", "metadata": {"doc_id": "d1"}}]
    )
    code, out, _ = _run_main(["--kb", "f" * 32, "search", "q"], fake)
    assert code == 0
    assert fake.sent["body"]["knowledge_id"] == "f" * 32


def test_main_global_json_before_subcommand():
    fake = _kbs_client()
    code, out, _ = _run_main(["--json", "kbs"], fake)
    assert code == 0
    assert json.loads(out)[0]["id"] == "id-1"


def test_main_search_uses_default_kb_from_env(monkeypatch):
    monkeypatch.setenv("PAIRAG_KB", "f" * 32)
    fake = _search_client(
        [{"content": "hi", "score": 0.4, "title": "T", "metadata": {"doc_id": "d1"}}]
    )
    code, out, _ = _run_main(["search", "q"], fake)
    assert code == 0
    assert fake.sent["body"]["knowledge_id"] == "f" * 32


def test_main_maps_pairag_error_to_exit_1():
    fake = _client(opener=None)

    def boom(method, path, params=None, body=None):
        raise pairag.PairagError("kaboom")

    fake._request = boom
    code, out, err = _run_main(["kbs"], fake)
    assert code == 1
    assert "kaboom" in err
