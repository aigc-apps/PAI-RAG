from __future__ import annotations

import asyncio
import base64
import datetime
import json
import os
import shlex
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from agent.tools.scope import get_current_tool_scope


@dataclass
class _SandboxSession:
    handle: Any
    last_used: float
    last_checked: float = 0.0
    # Epoch seconds when the STS creds last injected into this sandbox expire
    # (None when no expiring creds were injected). Drives pre-expiry re-inject.
    env_expires_at: Optional[float] = None


class SandboxUnavailable(RuntimeError):
    """The cached sandbox handle no longer points to a live sandbox instance."""


class ScopedSandboxProvider:
    """Session cache shared by sandbox providers.

    Sessions are isolated by user by default, with optional tenant/conversation
    scoping. Creation is lazy: first tool call creates the sandbox; later
    calls reuse the warm session until `session_idle_seconds` expires.
    """

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.template_name = str(settings.get("template_name") or "")
        self.template_type = str(settings.get("template_type") or "CodeInterpreter")
        self.idle_timeout_seconds = int(settings.get("idle_timeout_seconds") or 600)
        self.session_idle_seconds = int(
            settings.get("session_idle_seconds") or self.idle_timeout_seconds
        )
        self.isolation_scope = str(settings.get("isolation_scope") or "conversation")
        self.default_timeout_seconds = int(settings.get("timeout_seconds") or 60)
        self.cwd = str(settings.get("cwd") or "/home/user")
        self._sessions: Dict[str, _SandboxSession] = {}
        self._lock = threading.Lock()

    async def run_code(
        self,
        *,
        code: str,
        language: str,
        timeout: int,
        context_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._run_code_sync,
            self._scope_key(),
            code,
            language,
            timeout or self.default_timeout_seconds,
            context_id,
            cwd or self.cwd,
        )

    async def run_command(
        self,
        *,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._run_command_sync,
            self._scope_key(),
            command,
            timeout or self.default_timeout_seconds,
            cwd or self.cwd,
        )

    def _run_code_sync(
        self,
        scope_key: str,
        code: str,
        language: str,
        timeout: int,
        context_id: Optional[str],
        cwd: str,
    ) -> Dict[str, Any]:
        handle = self._ensure_sandbox(scope_key)
        return self._execute_code(
            handle,
            scope_key=scope_key,
            code=code,
            language=language,
            timeout=timeout,
            context_id=context_id,
            cwd=cwd,
        )

    def _run_command_sync(
        self,
        scope_key: str,
        command: str,
        timeout: int,
        cwd: str,
    ) -> Dict[str, Any]:
        handle = self._ensure_sandbox(scope_key)
        return self._execute_command(
            handle,
            scope_key=scope_key,
            command=command,
            timeout=timeout,
            cwd=cwd,
        )

    def _ensure_sandbox(self, scope_key: str):
        now = time.monotonic()
        with self._lock:
            self._reap_idle_locked(now)
            session = self._sessions.get(scope_key)
            if session is None:
                session = _SandboxSession(handle=self._create_sandbox(scope_key), last_used=now)
                self._sessions[scope_key] = session
            session.last_used = now
            return session.handle

    def _reap_idle_locked(self, now: float) -> None:
        expired = [
            key for key, session in self._sessions.items()
            if now - session.last_used > self.session_idle_seconds
        ]
        for key in expired:
            session = self._sessions.pop(key)
            try:
                self._stop_sandbox(session.handle)
            except Exception as exc:
                logger.debug(f"stop idle sandbox session {key} failed: {exc}")

    def _scope_key(self) -> str:
        scope = get_current_tool_scope()
        if self.isolation_scope == "conversation" and scope.conversation_id:
            return _append_agent_scope(f"conversation:{scope.conversation_id}", scope)
        if self.isolation_scope == "tenant":
            tenant_id = scope.metadata.get("tenant_id") or scope.metadata.get("tenant")
            if tenant_id:
                return _append_agent_scope(f"tenant:{tenant_id}", scope)
        if scope.user_id:
            return _append_agent_scope(f"user:{scope.user_id}", scope)
        if scope.conversation_id:
            return _append_agent_scope(f"conversation:{scope.conversation_id}", scope)
        return _append_agent_scope("anonymous", scope)

    def _create_sandbox(self, scope_key: str):
        raise NotImplementedError

    def _execute_code(
        self,
        handle,
        *,
        scope_key: str,
        code: str,
        language: str,
        timeout: int,
        context_id: Optional[str],
        cwd: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _execute_command(
        self,
        handle,
        *,
        scope_key: str,
        command: str,
        timeout: int,
        cwd: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _stop_sandbox(self, handle) -> None:
        raise NotImplementedError


class AgentRunRestSandboxProvider(ScopedSandboxProvider):
    """REST gateway provider for Alibaba Cloud AgentRun sandboxes.

    This is the production-friendly path: the agent service calls a small
    internal gateway (or the AgentRun data endpoint directly), which handles
    Alibaba Cloud auth/signing, AgentRun sandbox creation, dynamic OSS/NAS
    mounts, and hard cleanup.

    Required settings: template_name, api_key, account_id. The gateway endpoint
    is optional; when omitted it is derived as
    `https://{account_id}.agentrun-data.{region}.aliyuncs.com`.

    Default gateway contract:
      POST {endpoint}/sandboxes
      POST {endpoint}/sandboxes/{sandbox_id}/contexts/execute
      POST {endpoint}/sandboxes/{sandbox_id}/processes/cmd
      POST {endpoint}/sandboxes/{sandbox_id}/stop
    """

    def __init__(self, settings: Dict[str, Any]):
        super().__init__(settings)
        self.api_key = _setting_or_env(settings, "api_key", "api_key_env")
        self.api_key_header = str(settings.get("api_key_header") or "X-API-Key")
        self.parent_id = _setting_or_env(settings, "account_id", "account_id_env")
        # Gateway endpoint is optional: if not set, derive the AgentRun data
        # endpoint from the account id and region.
        endpoint = str(settings.get("endpoint") or "").rstrip("/")
        if not endpoint:
            region = str(settings.get("region") or settings.get("region_id") or "cn-hangzhou")
            endpoint = f"https://{self.parent_id}.agentrun-data.{region}.aliyuncs.com"
        self.endpoint = endpoint
        self.request_timeout = float(settings.get("request_timeout_seconds") or 60)
        self.create_path = str(settings.get("create_path") or "/sandboxes")
        self.execute_path = str(
            settings.get("execute_path") or "/sandboxes/{sandbox_id}/contexts/execute"
        )
        self.cmd_path = str(
            settings.get("cmd_path") or "/sandboxes/{sandbox_id}/processes/cmd"
        )
        self.stop_path = str(settings.get("stop_path") or "/sandboxes/{sandbox_id}/stop")
        self.health_path = str(settings.get("health_path") or "")
        self.health_check_interval_seconds = int(
            settings.get("health_check_interval_seconds") or 0
        )
        # Per-user /mnt/user NAS mount. The remote path is templated from the
        # tool scope so each user lands in an isolated NAS subdirectory.
        nas_user_cfg = settings.get("nas_config") or {}
        if not isinstance(nas_user_cfg, dict):
            nas_user_cfg = {}
        self.nas_user_id = int(nas_user_cfg.get("user_id") or 1000)
        self.nas_group_id = int(nas_user_cfg.get("group_id") or 1000)
        self.nas_user_server_addr = str(
            nas_user_cfg.get("user_server_addr") or nas_user_cfg.get("server_addr") or ""
        )
        self.nas_user_remote_path_template = str(
            nas_user_cfg.get("user_remote_path_template") or "/users/{user_id}"
        )
        self.nas_user_read_only = bool(nas_user_cfg.get("user_read_only", False))
        # Read-only code layer at /mnt/code: a single NAS export whose
        # subdirectories are source repositories the agent can explore when the
        # knowledge base can't answer. Enabled purely by configuring
        # `code_server_addr`; empty => the layer (mount + AGENT_CODE_PATH) is off.
        self.nas_code_server_addr = str(nas_user_cfg.get("code_server_addr") or "")
        self.nas_code_remote_path = str(nas_user_cfg.get("code_remote_path") or "/code")
        self.nas_code_read_only = bool(nas_user_cfg.get("code_read_only", True))
        # Runtime env-var contract injected via the platform `envs` field. Only
        # the AGENT_* marker vars + session/user ids are injected here; PATH and
        # PYTHONPATH stay image-side (flat `envs` map cannot interpolate).
        self.inject_env_contract = bool(settings.get("inject_env_contract", True))
        extra_envs = settings.get("extra_envs") or {}
        self.extra_envs = {str(k): str(v) for k, v in extra_envs.items()} if isinstance(extra_envs, dict) else {}
        self._async_lock = asyncio.Lock()

    async def run_code(
        self,
        *,
        code: str,
        language: str,
        timeout: int,
        context_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        scope_key = self._scope_key()
        handle = await self._ensure_sandbox_async(scope_key)
        try:
            return await self._execute_code_async(
                handle,
                scope_key=scope_key,
                code=code,
                language=language,
                timeout=timeout or self.default_timeout_seconds,
                context_id=context_id,
                cwd=cwd or self.cwd,
            )
        except SandboxUnavailable:
            await self._discard_sandbox_async(scope_key, handle)
            handle = await self._ensure_sandbox_async(scope_key)
            return await self._execute_code_async(
                handle,
                scope_key=scope_key,
                code=code,
                language=language,
                timeout=timeout or self.default_timeout_seconds,
                context_id=context_id,
                cwd=cwd or self.cwd,
            )

    async def run_command(
        self,
        *,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        scope_key = self._scope_key()
        handle = await self._ensure_sandbox_async(scope_key)
        try:
            return await self._execute_command_async(
                handle,
                scope_key=scope_key,
                command=command,
                timeout=timeout or self.default_timeout_seconds,
                cwd=cwd or self.cwd,
            )
        except SandboxUnavailable:
            await self._discard_sandbox_async(scope_key, handle)
            handle = await self._ensure_sandbox_async(scope_key)
            return await self._execute_command_async(
                handle,
                scope_key=scope_key,
                command=command,
                timeout=timeout or self.default_timeout_seconds,
                cwd=cwd or self.cwd,
            )

    async def _ensure_sandbox_async(self, scope_key: str) -> str:
        now = time.monotonic()
        async with self._async_lock:
            await self._reap_idle_async(now)
            session = self._sessions.get(scope_key)
            if session is None:
                session = await self._new_session_async(scope_key, now)
            else:
                if await self._should_check_health_async(session, now):
                    if await self._sandbox_alive_async(session.handle):
                        session.last_checked = now
                    else:
                        session = await self._new_session_async(scope_key, now)
                # A reused, still-alive sandbox keeps whatever STS creds were last
                # written into it. Re-inject fresh creds before they expire so a
                # long-running conversation never hits InvalidSecurityToken.Expired.
                await self._maybe_refresh_env_async(session, scope_key)
            session.last_used = now
            return str(session.handle)

    async def _new_session_async(self, scope_key: str, now: float) -> "_SandboxSession":
        session = _SandboxSession(
            handle=await self._create_sandbox_async(scope_key),
            last_used=now,
            last_checked=now,
            env_expires_at=_env_contract_expiry(_build_env_contract(
                self, get_current_tool_scope(), scope_key)),
        )
        self._sessions[scope_key] = session
        return session

    async def _maybe_refresh_env_async(self, session: "_SandboxSession", scope_key: str) -> None:
        # Already holding creds that are valid well past the refresh margin → nothing
        # to do. (env_expires_at is None means the sandbox has no STS creds yet — a
        # sandbox created while the user was unbound — so we fall through and check
        # whether this turn's scope can now supply them.)
        if (session.env_expires_at is not None
                and time.time() < session.env_expires_at - _ENV_REFRESH_MARGIN_SECONDS):
            return
        # Rebuild from the current turn's scope, which carries a freshly minted
        # token (the builder re-assumes each turn). new_expiry is None when the
        # scope still has no expiring creds (user hasn't authorized yet) — leave the
        # sandbox as-is rather than run a pointless bootstrap.
        env_contract = _build_env_contract(self, get_current_tool_scope(), scope_key)
        new_expiry = _env_contract_expiry(env_contract)
        if new_expiry is None:
            return
        # Covers both cases: creds nearing expiry are replaced, and a sandbox that
        # started credential-less gets its first injection once the user authorizes
        # mid-session (so the very next turn's aliyun calls just work).
        first_injection = session.env_expires_at is None
        await self._bootstrap_env_async(str(session.handle), env_contract)
        session.env_expires_at = new_expiry
        logger.info(
            "sandbox STS creds {} sandboxId={} scope={}",
            "injected" if first_injection else "re-injected", session.handle, scope_key,
        )

    async def _reap_idle_async(self, now: float) -> None:
        expired = [
            key for key, session in self._sessions.items()
            if now - session.last_used > self.session_idle_seconds
        ]
        for key in expired:
            session = self._sessions.pop(key)
            try:
                await self._stop_sandbox_async(session.handle)
            except Exception as exc:
                logger.debug(f"stop idle sandbox session {key} failed: {exc}")

    async def _create_sandbox_async(self, scope_key: str) -> str:
        if not self.endpoint:
            raise RuntimeError("sandbox REST provider requires settings.endpoint")
        if not self.parent_id:
            raise RuntimeError(
                "sandbox REST provider requires Alibaba Cloud account id: "
                "set settings.account_id or AGENTRUN_ACCOUNT_ID"
            )
        scope = get_current_tool_scope()
        env_contract = _build_env_contract(self, scope, scope_key)
        payload = _compact_dict({
            "templateName": self.template_name,
            "templateType": self.template_type or None,
            "sandboxId": _scoped_sandbox_id(self.settings, scope_key),
            "nasConfig": _build_nas_config(self, scope, scope_key),
            # NOTE: AgentRun's CreateSandbox input has no `envs` field — this is
            # ignored by the platform and kept only for forward-compat. The env
            # contract is delivered by _bootstrap_env_async below (writes
            # ~/.bash_env + ~/.aliyun/config.json inside the started sandbox).
            "envs": env_contract,
        })
        try:
            body = await self._request_async("POST", self.create_path, json=payload)
        except SandboxUnavailable as exc:
            # Create returned 404/410 — almost always a config mismatch, not a
            # stale-cache situation. Surface the gateway body + the values we
            # sent so the operator can see which of template_name / account_id
            # / region to check (a "template not found" here usually means the
            # template doesn't exist under this account_id in this region).
            raise RuntimeError(
                f"sandbox create failed (template={self.template_name!r}, "
                f"account_id={self.parent_id!r}): {exc}. "
                f"Verify settings.template_name and settings.account_id match a "
                f"template registered in AgentRun under that account (and region)."
            ) from exc
        sandbox_id = _extract_sandbox_id(body)
        if not sandbox_id:
            raise RuntimeError("sandbox REST create response missing sandboxId")
        logger.info(
            "sandbox REST created sandboxId={} scope={} skill_mounts={}",
            sandbox_id,
            scope_key,
            len(scope.skill_mounts),
        )
        await self._bootstrap_env_async(str(sandbox_id), env_contract)
        return str(sandbox_id)

    async def _bootstrap_env_async(
        self, sandbox_id: str, env_contract: Optional[Dict[str, str]]
    ) -> None:
        """Deliver the runtime env contract into a freshly-created sandbox.

        AgentRun's CreateSandbox has no `envs` field, so the create-time contract
        never reaches the FC container. Instead we run one command right after
        create that writes the vars into ~/.bash_env (sourced by every
        non-interactive shell via the image's BASH_ENV hook) and, when Aliyun STS
        session creds are present, an `authz` StsToken profile into
        ~/.aliyun/config.json so the baked aliyun CLI authenticates flag-free.
        Best-effort: a failure here leaves the sandbox usable for non-credentialed
        work, so we log and continue rather than failing sandbox creation."""
        command = _env_bootstrap_command(env_contract)
        if not command:
            logger.info("sandbox env bootstrap: no contract to inject sandboxId={}", sandbox_id)
            return
        # Whether the aliyun StsToken profile will actually be written mirrors the
        # `if ak and sk and tok` guard in _ENV_BOOTSTRAP_PY: aliyun_sts_profile=False
        # here is the reason a sandbox reports "profile default is not configure yet".
        # Log key NAMES only (values are secret) so this is safe and greppable.
        contract = env_contract or {}
        has_sts = all(
            contract.get(k)
            for k in (
                "ALIBABACLOUD_ACCESS_KEY_ID",
                "ALIBABACLOUD_ACCESS_KEY_SECRET",
                "ALIBABACLOUD_SECURITY_TOKEN",
            )
        )
        logger.info(
            "sandbox env bootstrap: injecting {} vars, aliyun_sts_profile={} "
            "sandboxId={} keys={}",
            len(contract), has_sts, sandbox_id, sorted(contract),
        )
        try:
            await self._request_async(
                "POST",
                self.cmd_path.format(sandbox_id=sandbox_id),
                json=_compact_dict({"command": command, "cwd": self.cwd}),
                sensitive=True,
            )
        except Exception as exc:  # pragma: no cover - network/gateway failure path
            logger.warning(
                "sandbox env bootstrap failed sandboxId={}: {}", sandbox_id, exc
            )

    async def _should_check_health_async(self, session: _SandboxSession, now: float) -> bool:
        return bool(
            self.health_path
            and self.health_check_interval_seconds > 0
            and now - session.last_checked >= self.health_check_interval_seconds
        )

    async def _sandbox_alive_async(self, handle) -> bool:
        try:
            await self._request_async(
                "GET",
                self.health_path.format(sandbox_id=handle),
                json=None,
            )
            return True
        except SandboxUnavailable:
            return False

    async def _discard_sandbox_async(self, scope_key: str, handle: str) -> None:
        async with self._async_lock:
            session = self._sessions.get(scope_key)
            if session is not None and str(session.handle) == str(handle):
                self._sessions.pop(scope_key, None)

    async def _execute_code_async(
        self,
        handle,
        *,
        scope_key: str,
        code: str,
        language: str,
        timeout: int,
        context_id: Optional[str],
        cwd: str,
    ) -> Dict[str, Any]:
        path = self.execute_path.format(sandbox_id=handle)
        body = await self._request_async(
            "POST",
            path,
            json={
                "scope_key": scope_key,
                "code": code,
                "language": language,
                "timeout": max(1, min(int(timeout), 30)),
                "contextId": context_id,
                "cwd": cwd,
            },
        )
        return _normalize_execution_result(body)

    async def _execute_command_async(
        self,
        handle,
        *,
        scope_key: str,
        command: str,
        timeout: int,
        cwd: str,
    ) -> Dict[str, Any]:
        # AgentRun's processes/cmd route takes only {command, cwd}; the gateway
        # enforces a hard 30s ceiling regardless of the requested timeout.
        path = self.cmd_path.format(sandbox_id=handle)
        body = await self._request_async(
            "POST",
            path,
            json=_compact_dict({
                "command": command,
                "cwd": cwd,
            }),
        )
        return _normalize_command_result(body)

    async def _stop_sandbox_async(self, handle) -> None:
        await self._request_async("POST", self.stop_path.format(sandbox_id=handle), json=None)

    async def _request_async(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]],
        sensitive: bool = False,
    ):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.api_key_header] = self.api_key
        if self.parent_id:
            headers["X-Acs-Parent-Id"] = self.parent_id
        url = f"{self.endpoint}{path if path.startswith('/') else '/' + path}"
        # `sensitive` bodies (the env-bootstrap write) carry credentials inside a
        # non-secret-named `command` string that _mask_body cannot see into, so
        # redact the whole body rather than log it.
        logged_body = (
            "<redacted>" if sensitive else (_mask_body(json) if json is not None else None)
        )
        logger.info(
            "sandbox REST request {} {} headers={} body={}",
            method,
            url,
            _mask_headers(headers),
            logged_body,
        )
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.request(method, url, headers=headers, json=json)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body_text = _response_text(exc.response)
                logger.warning(
                    "sandbox REST request failed status={} method={} url={} headers={} body={}",
                    exc.response.status_code,
                    method,
                    url,
                    _mask_headers(headers),
                    body_text,
                )
                if exc.response.status_code in {404, 410}:
                    raise SandboxUnavailable(
                        f"cached sandbox is no longer available: "
                        f"{method} {path} -> {exc.response.status_code} {body_text[:300]}"
                    ) from exc
                raise RuntimeError(
                    f"sandbox REST {method} {path} failed: "
                    f"{exc.response.status_code} {body_text[:300]}"
                ) from exc
            if not resp.content:
                return {}
            body = resp.json()
            logger.debug(
                "sandbox REST response status={} method={} url={} body={}",
                resp.status_code,
                method,
                url,
                _mask_body(body),
            )
            return body

    def _create_sandbox(self, scope_key: str):
        raise RuntimeError("agentrun_rest provider is async-only")

    def _execute_code(
        self,
        handle,
        *,
        scope_key: str,
        code: str,
        language: str,
        timeout: int,
        context_id: Optional[str],
        cwd: str,
    ) -> Dict[str, Any]:
        raise RuntimeError("agentrun_rest provider is async-only")

    def _execute_command(
        self,
        handle,
        *,
        scope_key: str,
        command: str,
        timeout: int,
        cwd: str,
    ) -> Dict[str, Any]:
        raise RuntimeError("agentrun_rest provider is async-only")

    def _stop_sandbox(self, handle) -> None:
        raise RuntimeError("agentrun_rest provider is async-only")


class AgentRunSdkSandboxProvider(ScopedSandboxProvider):
    """Optional SDK-backed AgentRun provider.

    Kept for development/prototyping. Production deployments should prefer
    `agentrun_rest` so the agent service does not carry cloud SDK dependencies
    or signing logic.
    """

    def _create_sandbox(self, scope_key: str):
        try:
            from agentrun.sandbox import Sandbox, TemplateType
            from agentrun.sandbox.model import NASConfig, OSSMountConfig, PolarFsConfig
            from agentrun.utils.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "agentrun-sdk is required for sandbox provider 'agentrun'; "
                "install it with `pip install agentrun-sdk`"
            ) from exc

        if not self.template_name:
            raise RuntimeError("sandbox provider requires settings.template_name")

        template_type = getattr(TemplateType, "CODE_INTERPRETER")
        if self.template_type in {"Browser", "AllInOne", "CustomImage"}:
            template_type = TemplateType(self.template_type)

        return Sandbox.create(
            template_type=template_type,
            template_name=self.template_name,
            sandbox_idle_timeout_seconds=self.idle_timeout_seconds,
            sandbox_id=_scoped_sandbox_id(self.settings, scope_key),
            oss_mount_config=_model_or_none(OSSMountConfig, self.settings.get("oss_mount_config")),
            nas_config=_model_or_none(NASConfig, self.settings.get("nas_config")),
            polar_fs_config=_model_or_none(PolarFsConfig, self.settings.get("polar_fs_config")),
            config=Config(
                access_key_id=_setting_or_env(self.settings, "access_key_id", "access_key_id_env"),
                access_key_secret=_setting_or_env(
                    self.settings, "access_key_secret", "access_key_secret_env"
                ),
                security_token=_setting_or_env(
                    self.settings, "security_token", "security_token_env"
                ),
                account_id=_setting_or_env(self.settings, "account_id", "account_id_env"),
                region_id=self.settings.get("region") or self.settings.get("region_id"),
                control_endpoint=self.settings.get("control_endpoint") or None,
                data_endpoint=self.settings.get("data_endpoint") or None,
            ),
        )

    def _execute_code(
        self,
        handle,
        *,
        scope_key: str,
        code: str,
        language: str,
        timeout: int,
        context_id: Optional[str],
        cwd: str,
    ) -> Dict[str, Any]:
        if context_id:
            result = handle.context.execute(
                code=code,
                context_id=context_id,
                language=language,
                timeout=timeout,
            )
        else:
            context = handle.context.create(language=language, cwd=cwd)
            try:
                result = context.execute(code=code, timeout=timeout)
            finally:
                try:
                    context.delete()
                except Exception as exc:
                    logger.debug(f"delete sandbox context failed: {exc}")
        return _normalize_execution_result(result)

    def _execute_command(
        self,
        handle,
        *,
        scope_key: str,
        command: str,
        timeout: int,
        cwd: str,
    ) -> Dict[str, Any]:
        result = handle.commands.run(command=command, timeout=timeout, cwd=cwd)
        return _normalize_command_result(result)

    def _stop_sandbox(self, handle) -> None:
        handle.stop()


def _normalize_execution_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(result.get("data"), dict):
        result = result["data"]
    if isinstance(result.get("results"), list):
        stdout_parts = []
        stderr_parts = []
        exit_code = 0
        for item in result["results"]:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            text = item.get("text")
            if item_type in {"stdout", "result"} and text is not None:
                stdout_parts.append(str(text))
            elif item_type in {"stderr", "error"} and text is not None:
                stderr_parts.append(str(text))
                exit_code = 1
            elif item_type == "endOfExecution" and item.get("status") not in {None, "ok"}:
                exit_code = 1
        return {
            "stdout": "\n".join(stdout_parts),
            "stderr": "\n".join(stderr_parts),
            "exit_code": exit_code,
            "artifacts": result.get("artifacts"),
            "context_id": result.get("contextId"),
            "raw": result,
        }
    return {
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
        "exit_code": result.get("exit_code", result.get("exitCode", 0)),
        "artifacts": result.get("artifacts"),
        "raw": result,
    }


def _normalize_command_result(result: Any) -> Dict[str, Any]:
    """Normalize a processes/cmd result from either the REST gateway or the SDK.

    REST shape: ``{executionId, status, result: {exitCode, stdout, stderr,
    cwd, executionTimeMs}, executionTimeMs}``. SDK shape: a CommandResult-like
    object (pydantic model or dict) with ``exit_code/exitCode, stdout, stderr``.
    """
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    if not isinstance(result, dict):
        return {"stdout": "", "stderr": str(result), "exit_code": 1, "raw": result}
    status = result.get("status")
    inner = result.get("result")
    if isinstance(inner, dict):
        result = inner
    exit_code = result.get("exit_code", result.get("exitCode", 0))
    # A non-terminal status (e.g. "timeout", "error") with a zero exit code must
    # still surface as a failure to the model.
    if status not in (None, "", "completed", "ok", "success") and not exit_code:
        exit_code = 1
    return {
        "stdout": str(result.get("stdout") or ""),
        "stderr": str(result.get("stderr") or ""),
        "exit_code": exit_code,
        "raw": result,
    }


def _extract_sandbox_id(body: Dict[str, Any]) -> Optional[str]:
    sandbox_id = body.get("sandbox_id") or body.get("id") or body.get("sandboxId")
    if sandbox_id:
        return str(sandbox_id)
    data = body.get("data")
    if isinstance(data, dict):
        sandbox_id = data.get("sandbox_id") or data.get("id") or data.get("sandboxId")
        if sandbox_id:
            return str(sandbox_id)
    return None


def _model_or_none(model_cls, value):
    if not value:
        return None
    if isinstance(value, model_cls):
        return value
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(value)
    return model_cls(**value)


def _setting_or_env(settings: Dict[str, Any], direct_key: str, env_key: str) -> Optional[str]:
    direct = settings.get(direct_key)
    if direct:
        return str(direct)
    env_name = settings.get(env_key)
    if env_name:
        return os.environ.get(str(env_name)) or None
    return None


def _mask_headers(headers: Dict[str, str]) -> Dict[str, str]:
    masked = dict(headers)
    for key in ("Authorization", "X-API-Key", "X-Acs-Security-Token"):
        value = masked.get(key)
        if value:
            masked[key] = _mask_secret(value)
    return masked


def _mask_secret(value: str) -> str:
    if len(value) <= 12:
        return "********"
    return f"{value[:8]}...{value[-4:]}"


def _response_text(response: httpx.Response) -> str:
    try:
        return response.text[:1000]
    except Exception:
        return ""


def _compact_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


# Case-insensitive markers for keys whose values are secrets and must never be
# logged in cleartext. Matched against the alphanumeric-normalized key, so this
# covers apiKey/api_key, accessKeySecret, securityToken, AND the injected sandbox
# session vars ALIBABACLOUD_ACCESS_KEY_ID/_SECRET / ALIBABACLOUD_SECURITY_TOKEN.
_SECRET_KEY_MARKERS = ("secret", "token", "password", "accesskey", "apikey")


def _is_secret_key(key: Any) -> bool:
    normalized = "".join(ch for ch in str(key).lower() if ch.isalnum())
    return any(marker in normalized for marker in _SECRET_KEY_MARKERS)


def _mask_body(value: Any) -> Any:
    if isinstance(value, dict):
        masked = {}
        for key, item in value.items():
            if _is_secret_key(key):
                masked[key] = _mask_secret(str(item))
            else:
                masked[key] = _mask_body(item)
        return masked
    if isinstance(value, list):
        return [_mask_body(item) for item in value]
    return value


def _non_empty_mount_config(value):
    if not value:
        return None
    if not isinstance(value, dict):
        return value
    mount_points = value.get("mount_points")
    if mount_points is None:
        mount_points = value.get("mountPoints")
    if isinstance(mount_points, list) and not mount_points:
        return None
    return value


def _append_agent_scope(base: str, scope) -> str:
    parts = [base]
    if scope.agent_id:
        parts.append(f"agent:{scope.agent_id}")
    if scope.skill_fingerprint and scope.skill_fingerprint != "none":
        parts.append(f"skills:{scope.skill_fingerprint}")
    return ":".join(parts)


def _build_nas_config(provider: "AgentRunRestSandboxProvider", scope, scope_key: str) -> Optional[Dict[str, Any]]:
    """Build the nasConfig payload: per-skill read-only mounts under
    /mnt/skills/<id>, a per-user read-write mount at /mnt/user, and (when
    configured) a read-only code layer at /mnt/code. All mountPoints share
    userId/groupId (default 1000). Returns None when no mountPoints apply so the
    field is omitted from the create payload."""
    mount_points: List[Dict[str, Any]] = []
    for mount in scope.skill_mounts:
        if not isinstance(mount, dict):
            continue
        nas = mount.get("nas")
        if not isinstance(nas, dict) or not nas.get("serverAddr"):
            continue
        mount_points.append({
            "serverAddr": nas["serverAddr"],
            "mountDir": nas.get("mountDir"),
            "readOnly": bool(nas.get("readOnly", True)),
        })
    # Per-user /mnt/user mount: remote path is templated from the user id so
    # each user lands in an isolated NAS subdirectory.
    if provider.nas_user_server_addr:
        user_id = scope.user_id or _scope_fallback_id(scope_key)
        remote_path = provider.nas_user_remote_path_template.format(user_id=user_id)
        mount_points.append({
            "serverAddr": _join_nas_server_addr(provider.nas_user_server_addr, remote_path),
            "mountDir": "/mnt/user",
            "readOnly": provider.nas_user_read_only,
        })
    # Read-only code layer at /mnt/code (single shared export; repos are its
    # subdirectories). Configured => mounted for everyone; omitted otherwise.
    if provider.nas_code_server_addr:
        mount_points.append({
            "serverAddr": _join_nas_server_addr(
                provider.nas_code_server_addr, provider.nas_code_remote_path),
            "mountDir": "/mnt/code",
            "readOnly": provider.nas_code_read_only,
        })
    if not mount_points:
        return None
    return {
        "userId": provider.nas_user_id,
        "groupId": provider.nas_group_id,
        "mountPoints": mount_points,
    }


def _build_env_contract(provider: "AgentRunRestSandboxProvider", scope, scope_key: str) -> Optional[Dict[str, str]]:
    """Inject the AGENT_* runtime contract via the platform `envs` field. PATH
    and PYTHONPATH are intentionally not set here (the flat string map cannot
    interpolate); they are baked into the sandbox image."""
    if not provider.inject_env_contract:
        return None
    envs: Dict[str, str] = {
        "AGENT_SYSTEM_PATH": "/mnt/system",
        "AGENT_SKILL_PATH": "/mnt/skills",
        "AGENT_USER_PATH": "/mnt/user",
        "AGENT_USER_ID": str(scope.user_id or ""),
        "AGENT_SESSION_ID": scope_key,
    }
    # Only advertise the code layer when it is actually mounted, so the agent
    # never sees AGENT_CODE_PATH for a /mnt/code that does not exist.
    if provider.nas_code_server_addr:
        envs["AGENT_CODE_PATH"] = "/mnt/code"
    envs.update(provider.extra_envs)
    # Per-user Aliyun session credentials carried on the scope. The secret + token
    # keys are redacted in create-payload logs by _mask_body / _is_secret_key.
    extra = scope.metadata.get("aliyun_sandbox_env")
    if isinstance(extra, dict):
        envs.update({str(k): str(v) for k, v in extra.items()})
    return envs


# Marker block bounding our exports in ~/.bash_env so a re-bootstrap on sandbox
# recreate replaces (rather than appends to) the previous contract.
_BASH_ENV_BEGIN = "# >>> agent env contract >>>"
_BASH_ENV_END = "# <<< agent env contract <<<"

# Re-inject STS creds this many seconds before the injected token expires.
_ENV_REFRESH_MARGIN_SECONDS = 300


def _parse_iso_expiry(value: Any) -> Optional[float]:
    """Parse an ISO8601 timestamp (e.g. AssumeRole's '2026-07-08T09:20:00Z')
    into epoch seconds. Returns None on anything unparseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()


def _env_contract_expiry(env_contract: Optional[Dict[str, str]]) -> Optional[float]:
    """Epoch expiry of the STS creds in an env contract, or None if it carries
    no expiring session credentials."""
    if not env_contract:
        return None
    return _parse_iso_expiry(env_contract.get("ALIBABACLOUD_SESSION_EXPIRATION"))


def _env_bootstrap_command(env_contract: Optional[Dict[str, str]]) -> Optional[str]:
    """Build a single shell command that installs the env contract inside the
    sandbox. AgentRun's CreateSandbox has no `envs` field, so the contract is
    delivered post-create by writing ~/.bash_env (picked up by the image's
    BASH_ENV hook on every non-interactive shell) and, when STS session creds are
    present, an `authz` StsToken profile in ~/.aliyun/config.json (the CLI reads
    the `current` profile, ignoring env vars, so a file is the only reliable way).

    The whole payload is base64-wrapped so no credential appears as plaintext in
    the command string (which is also logged with sensitive=True redaction)."""
    if not env_contract:
        return None
    py = _ENV_BOOTSTRAP_PY.replace("__ENVS_JSON__", json.dumps(json.dumps(env_contract)))
    blob = base64.b64encode(py.encode("utf-8")).decode("ascii")
    return f"printf %s {shlex.quote(blob)} | base64 -d | python3 -"


# Runs inside the sandbox (base64-piped to `python3 -`). Writes the shell env
# contract and merges the Aliyun CLI StsToken profile. Kept dependency-free
# (stdlib only) since it runs against the sandbox's own interpreter.
_ENV_BOOTSTRAP_PY = r'''
import json, os, shlex
envs = json.loads(__ENVS_JSON__)
home = os.path.expanduser("~")

# 1) Shell env for non-interactive bash (image sets BASH_ENV=~/.bash_env).
be = os.path.join(home, ".bash_env")
try:
    with open(be, "r", encoding="utf-8") as fh:
        text = fh.read()
except OSError:
    text = ""
begin, end = "# >>> agent env contract >>>", "# <<< agent env contract <<<"
if begin in text and end in text:
    text = text.split(begin)[0] + text.split(end, 1)[1]
block = [begin] + ["export %s=%s" % (k, shlex.quote(str(v))) for k, v in envs.items()] + [end]
text = (text.rstrip("\n") + "\n" if text.strip() else "") + "\n".join(block) + "\n"
with open(be, "w", encoding="utf-8") as fh:
    fh.write(text)

# 2) Aliyun CLI StsToken profile so `aliyun`/EAS work without flags.
ak = envs.get("ALIBABACLOUD_ACCESS_KEY_ID")
sk = envs.get("ALIBABACLOUD_ACCESS_KEY_SECRET")
tok = envs.get("ALIBABACLOUD_SECURITY_TOKEN")
if ak and sk and tok:
    cfgdir = os.path.join(home, ".aliyun")
    os.makedirs(cfgdir, exist_ok=True)
    cfgp = os.path.join(cfgdir, "config.json")
    try:
        with open(cfgp, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    profiles = [p for p in cfg.get("profiles", [])
                if isinstance(p, dict) and p.get("name") != "authz"]
    profiles.append({
        "name": "authz",
        "mode": "StsToken",
        "access_key_id": ak,
        "access_key_secret": sk,
        "sts_token": tok,
        "region_id": envs.get("ALIBABACLOUD_REGION_ID") or "cn-hangzhou",
    })
    cfg["profiles"] = profiles
    cfg["current"] = "authz"
    with open(cfgp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    os.chmod(cfgp, 0o600)
'''


def _join_nas_server_addr(server_addr: str, remote_path: str) -> str:
    """Combine a NAS server address and a remote path into a serverAddr value
    of the form '<server>:/<path>'. Accepts 'host:/', 'host:/<path>', or a bare
    host as the server_addr input."""
    remote = remote_path if remote_path.startswith("/") else "/" + remote_path
    if server_addr.endswith(":/"):
        return f"{server_addr}{remote.lstrip('/')}"
    if ":/" in server_addr:
        return f"{server_addr.rstrip('/')}{remote}"
    return f"{server_addr}:{remote}"


def _scope_fallback_id(scope_key: str) -> str:
    """Stable fallback identifier when scope.user_id is absent."""
    import hashlib
    return hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:16]


def _scoped_sandbox_id(settings: Dict[str, Any], scope_key: str) -> Optional[str]:
    prefix = settings.get("sandbox_id_prefix")
    if not prefix:
        return None
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in scope_key)
    return f"{prefix}-{safe}"[:128]


def make_sandbox_provider(agent_config) -> Optional[ScopedSandboxProvider]:
    if agent_config is None:
        return None
    caps = {cap.id: cap for cap in agent_config.capabilities}
    sandbox = caps.get("sandbox")
    if sandbox is None or not sandbox.enabled:
        return None
    providers = {provider.id: provider for provider in agent_config.providers}
    provider = providers.get("sandbox.default")
    if provider is None:
        return None
    settings = dict(provider.settings or {})
    provider_name = str(settings.get("provider") or "none")
    if provider_name == "agentrun_rest":
        # Only template, api_key, and account_id are required. The gateway
        # endpoint is optional and auto-derived from account_id + region when
        # not supplied.
        if not (
            settings.get("template_name")
            and _configured(settings, "api_key", "api_key_env")
            and _configured(settings, "account_id", "account_id_env")
        ):
            return None
        return AgentRunRestSandboxProvider(settings)
    if provider_name == "agentrun":
        if not settings.get("template_name"):
            return None
        required = (
            ("access_key_id", "access_key_id_env"),
            ("access_key_secret", "access_key_secret_env"),
            ("account_id", "account_id_env"),
        )
        if any(not _configured(settings, direct, env) for direct, env in required):
            return None
        return AgentRunSdkSandboxProvider(settings)
    return None


def _configured(settings: Dict[str, Any], direct_key: str, env_key: str) -> bool:
    if settings.get(direct_key):
        return True
    env_name = settings.get(env_key)
    return bool(env_name and os.environ.get(str(env_name)))
