"""Reactive aliyun authorization detection for sandbox command results.

When the agent runs the `aliyun` CLI in the sandbox and it fails because the
user isn't authorized (or the STS creds are broken/expired), we surface a
structured authorization notice so the frontend can render an inline "去授权"
card — instead of dumping raw `InvalidSecurityToken` stderr on the user. This is
deliberately *lazy*: it fires only when an actual aliyun command fails with a
credential-class error, never proactively, so ordinary conversations are
undisturbed.

Kept separate from `shell.py` so `code_interpreter` (or others) can reuse it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from agent.integrations import aliyun_sts
from agent.tools.artifacts import emit_tool_notice
from agent.tools.scope import get_current_tool_scope

# Matches `aliyun` only in *command position* — at the start of the line or
# right after a shell separator (`;`, `|`, `&`, `&&`, `||`, `(`, newline) — so
# `aliyun sts ...` and `cd /tmp && aliyun ...` fire, but `echo aliyun` (aliyun as
# a mere argument) does not. A loose match would surface the card for unrelated
# command failures; a leading `sudo`/env-prefix is a tolerated miss.
_ALIYUN_CMD_RE = re.compile(r"(?:^|\|\||&&|[;&|\n(])\s*aliyun(?=\s|$)")

_UNBOUND_HINT = (
    "当前用户尚未完成阿里云授权,已向其展示「去授权」卡片。授权是用户在自己阿里云账号里的"
    "一键 ROS 操作,你无法代为完成 —— 请勿尝试 `aliyun configure`、写 ~/.aliyun/config.json "
    "或以任何方式自行配置凭证。请暂停 aliyun 相关操作,待用户完成授权后再继续,不要反复重试。"
)
_BOUND_HINT = (
    "阿里云凭证已失效或无法续期,已向用户展示「重新校验/重新授权」卡片。凭证由平台自动注入,"
    "请勿尝试 `aliyun configure` 自行配置。请暂停操作,待用户处理后再重试。"
)
_PERMISSION_HINT = (
    "该操作被拒:当前授权角色缺少所需权限(角色仅授予 pai:*/eas:*)。这不是未授权问题,"
    "请勿要求重新授权;改用授权范围内的操作,或告知用户需要扩展角色策略。"
)


def _is_aliyun_command(command: str) -> bool:
    return bool(_ALIYUN_CMD_RE.search(command or ""))


def maybe_emit_aliyun_notice(command: str, result: Dict[str, Any]) -> Optional[str]:
    """Inspect a finished sandbox command result for an aliyun auth failure.

    On a **credential-class** failure of an `aliyun` command — and only when the
    current scope reports the aliyun_pai capability is available — emit an
    ``aliyun_authorization`` notice for the frontend and return a short guidance
    string to fold into the model-facing output. On a **permission-class**
    failure return an explanation but emit no card. Returns ``None`` otherwise.
    """
    if "error" in result:  # gateway/provider error, not an aliyun CLI result
        return None
    exit_code = result.get("exit_code", result.get("exitCode", 0))
    try:
        failed = int(exit_code) != 0
    except (TypeError, ValueError):
        failed = bool(exit_code)
    if not failed or not _is_aliyun_command(command):
        return None

    stderr = str(result.get("stderr") or "")
    stdout = str(result.get("stdout") or "")
    code, message = aliyun_sts.parse_cli_error(stderr or stdout)
    cls = aliyun_sts.classify_cli_error(code, message)

    if cls == "permission":
        return _PERMISSION_HINT
    if cls != "credential":
        return None

    meta = (get_current_tool_scope().metadata or {})
    if not meta.get("aliyun_authz_available"):
        # Feature not configured for this deployment/agent — nothing to offer.
        return None
    bound = bool(meta.get("aliyun_bound"))
    emit_tool_notice({
        "kind": "aliyun_authorization",
        "bound": bound,
        "error_code": code,
        # HITL: this notice halts the agent turn and hands control to the user
        # (see agent.py). Credential-class failures always need a human step.
        "interrupt": True,
    })
    return _BOUND_HINT if bound else _UNBOUND_HINT
