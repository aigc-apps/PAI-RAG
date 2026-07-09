"""app.extractors — file bytes → Markdown boundary. Offline: the markitdown
path is exercised with a fake module injected via sys.modules, so these run
without the `parsers` extra installed."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.extractors import (
    SUPPORTED_EXTENSIONS,
    TEXT_PASSTHROUGH,
    extract_to_markdown,
)


def test_text_passthrough_decodes_directly():
    assert extract_to_markdown("notes.md", b"# Title\nbody") == "# Title\nbody"
    assert extract_to_markdown("a.txt", "héllo".encode("utf-8")) == "héllo"


def test_passthrough_never_imports_markitdown(monkeypatch):
    # even with markitdown absent, .md/.txt still work
    monkeypatch.setitem(sys.modules, "markitdown", None)  # import -> ImportError
    assert extract_to_markdown("x.markdown", b"ok") == "ok"


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="unsupported file type"):
        extract_to_markdown("archive.zip", b"...")
    with pytest.raises(ValueError, match="unsupported file type"):
        extract_to_markdown("noext", b"...")


class _FakeResult:
    def __init__(self, text):
        self.text_content = text


class _FakeMarkItDown:
    last_path = None

    def convert(self, path):
        _FakeMarkItDown.last_path = path
        # prove the temp file exists with the right suffix while converting
        assert os.path.exists(path)
        assert path.endswith(".pdf")
        return _FakeResult("# Extracted\ncontent from pdf")


def _install_fake_markitdown(monkeypatch):
    import types

    mod = types.ModuleType("markitdown")
    mod.MarkItDown = _FakeMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", mod)


def test_markitdown_path_wires_and_cleans_up(monkeypatch):
    _install_fake_markitdown(monkeypatch)
    out = extract_to_markdown("report.pdf", b"%PDF-1.4 bytes")
    assert out == "# Extracted\ncontent from pdf"
    # temp file removed after extraction
    assert not os.path.exists(_FakeMarkItDown.last_path)


def test_markitdown_empty_result_raises(monkeypatch):
    import types

    class _Empty:
        def convert(self, path):
            return _FakeResult("   \n  ")

    mod = types.ModuleType("markitdown")
    mod.MarkItDown = _Empty
    monkeypatch.setitem(sys.modules, "markitdown", mod)
    with pytest.raises(ValueError, match="no extractable text"):
        extract_to_markdown("empty.docx", b"...")


def test_missing_extra_gives_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "markitdown", None)  # import -> ImportError
    with pytest.raises(ValueError, match="parsers.*extra"):
        extract_to_markdown("report.pdf", b"...")


def test_supported_extensions_superset_of_passthrough():
    assert TEXT_PASSTHROUGH <= SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS and ".docx" in SUPPORTED_EXTENSIONS
