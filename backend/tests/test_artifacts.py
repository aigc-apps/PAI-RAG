import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent.tools.artifacts import (
    ArtifactTokenError,
    FileArtifact,
    begin_artifact_capture,
    classify,
    drain_artifacts,
    emit_artifact,
    end_artifact_capture,
    sign_artifact_token,
    verify_artifact_token,
)

SECRET = "unit-test-secret"


def test_token_roundtrip():
    tok = sign_artifact_token(
        user_id="u1", conversation_id="c1", rel="out/report.md",
        mime="text/markdown", kind="markdown", size=42, secret=SECRET,
    )
    claim = verify_artifact_token(tok, secret=SECRET)
    assert claim.user_id == "u1"
    assert claim.conversation_id == "c1"
    assert claim.rel == "out/report.md"
    assert claim.mime == "text/markdown"
    assert claim.kind == "markdown"
    assert claim.size == 42


def test_token_tamper_rejected():
    tok = sign_artifact_token(
        user_id="u1", conversation_id="c1", rel="a.md",
        mime="text/markdown", kind="markdown", size=1, secret=SECRET,
    )
    with pytest.raises(ArtifactTokenError):
        verify_artifact_token(tok + "x", secret=SECRET)
    with pytest.raises(ArtifactTokenError):
        verify_artifact_token(tok, secret="wrong-secret")
    with pytest.raises(ArtifactTokenError):
        verify_artifact_token("not-a-token", secret=SECRET)


def test_empty_secret_fails_closed():
    with pytest.raises(ArtifactTokenError):
        sign_artifact_token(
            user_id="u", conversation_id="c", rel="a", mime="m", kind="file",
            size=0, secret="",
        )


@pytest.mark.parametrize(
    "name,expected_kind",
    [
        ("chart.png", "image"),
        ("photo.JPG", "image"),
        ("report.md", "markdown"),
        ("index.html", "html"),
        ("data.csv", "text"),
        ("notes.txt", "text"),
        ("archive.zip", "file"),
        ("model.bin", "file"),
        ("diagram.svg", "file"),  # SVG intentionally not previewed inline
    ],
)
def test_classify_kind(name, expected_kind):
    _mime, kind = classify(name)
    assert kind == expected_kind


def test_classify_mime():
    assert classify("a.png")[0] == "image/png"
    assert classify("a.md")[0] == "text/markdown"
    assert classify("a.html")[0] == "text/html"
    assert classify("a.unknownext")[0] == "application/octet-stream"


def test_contextvar_capture_drain():
    token = begin_artifact_capture()
    try:
        emit_artifact(FileArtifact(id="t1", name="a.md", mime="text/markdown", size=3, kind="markdown"))
        emit_artifact(FileArtifact(id="t2", name="b.png", mime="image/png", size=9, kind="image"))
        drained = drain_artifacts()
        assert [a.id for a in drained] == ["t1", "t2"]
        # Draining clears the buffer.
        assert drain_artifacts() == []
    finally:
        end_artifact_capture(token)


def test_emit_outside_capture_is_noop():
    # No active capture -> emit is silently dropped, no error.
    emit_artifact(FileArtifact(id="x", name="x", mime="m", size=0, kind="file"))
    assert drain_artifacts() == []
