import os
import sys
import asyncio
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.message import ToolCall
from agent.tools.artifacts import verify_artifact_token
from agent.tools.base import ToolBox
from agent.tools.builtin.publish_artifact import make_publish_artifact_tool
from agent.tools.scope import ToolScope

SECRET = "publish-test-secret"


class _FakeProvider:
    """Records commands and answers `stat -c %s` with a fixed size."""

    nas_user_remote_path_template = "/users/{user_id}"

    def __init__(self, size=123):
        self.size = size
        self.commands = []

    async def run_command(self, *, command, cwd=None, timeout=None):
        self.commands.append(command)
        if command.startswith("stat"):
            return {"stdout": str(self.size), "stderr": "", "exit_code": 0}
        return {"stdout": "", "stderr": "", "exit_code": 0}


def _settings(**over):
    base = dict(files_url_secret=SECRET, files_nas_local_root="", files_max_bytes=25 * 1024 * 1024)
    base.update(over)
    return SimpleNamespace(**base)


def _dispatch(tool, args, scope):
    box = ToolBox([tool])
    tc = ToolCall(id="c1", name="publish_artifact", arguments=args)
    return asyncio.run(box.dispatch(tc, scope=scope))


def test_publish_emits_artifact_with_valid_token():
    provider = _FakeProvider(size=57)
    tool = make_publish_artifact_tool(provider, _settings())
    res = _dispatch(
        tool,
        '{"path": "/mnt/user/report.md"}',
        ToolScope(user_id="u1", conversation_id="c1"),
    )
    assert res.ok
    assert res.files and len(res.files) == 1
    art = res.files[0]
    assert art["name"] == "report.md"
    assert art["kind"] == "markdown"
    assert art["size"] == 57
    claim = verify_artifact_token(art["id"], secret=SECRET)
    assert claim.user_id == "u1"
    assert claim.rel == "report.md"
    # The model-facing string carries no bytes, just a confirmation.
    assert "Published" in res.content
    # It stat'd the sandbox path.
    assert any("stat" in cmd and "/mnt/user/report.md" in cmd for cmd in provider.commands)


def test_publish_rejects_path_outside_mnt_user():
    provider = _FakeProvider()
    tool = make_publish_artifact_tool(provider, _settings())
    res = _dispatch(
        tool, '{"path": "/etc/passwd"}', ToolScope(user_id="u1", conversation_id="c1")
    )
    assert res.ok  # tool returns a normal string, not an exception
    assert res.files is None
    assert "must be a file under /mnt/user" in res.content


def test_publish_reports_missing_file():
    class _MissingProvider(_FakeProvider):
        async def run_command(self, *, command, cwd=None, timeout=None):
            return {"stdout": "", "stderr": "No such file", "exit_code": 1}

    tool = make_publish_artifact_tool(_MissingProvider(), _settings())
    res = _dispatch(
        tool, '{"path": "/mnt/user/ghost.png"}', ToolScope(user_id="u1", conversation_id="c1")
    )
    assert res.files is None
    assert "was not found" in res.content


def test_publish_without_user_scope_errors():
    tool = make_publish_artifact_tool(_FakeProvider(), _settings())
    res = _dispatch(tool, '{"path": "/mnt/user/a.md"}', ToolScope())
    assert res.files is None
    assert "no user in scope" in res.content
