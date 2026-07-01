from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from agent.tools.scope import get_current_tool_scope


@dataclass
class _SandboxSession:
    handle: Any
    last_used: float
    last_checked: float = 0.0


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

    def _stop_sandbox(self, handle) -> None:
        raise NotImplementedError


class AgentRunRestSandboxProvider(ScopedSandboxProvider):
    """REST gateway provider for Alibaba Cloud AgentRun sandboxes.

    This is the production-friendly path: the agent service calls a small
    internal gateway, while the gateway handles Alibaba Cloud auth/signing,
    AgentRun sandbox creation, dynamic OSS/NAS mounts, and hard cleanup.

    Default gateway contract:
      POST {endpoint}/sandboxes
      POST {endpoint}/sandboxes/{sandbox_id}/contexts/execute
      POST {endpoint}/sandboxes/{sandbox_id}/stop
    """

    def __init__(self, settings: Dict[str, Any]):
        super().__init__(settings)
        self.endpoint = str(settings.get("endpoint") or "").rstrip("/")
        self.api_key = _setting_or_env(settings, "api_key", "api_key_env")
        self.api_key_header = str(settings.get("api_key_header") or "X-API-Key")
        self.parent_id = _setting_or_env(settings, "account_id", "account_id_env")
        self.request_timeout = float(settings.get("request_timeout_seconds") or 60)
        self.create_path = str(settings.get("create_path") or "/sandboxes")
        self.execute_path = str(
            settings.get("execute_path") or "/sandboxes/{sandbox_id}/contexts/execute"
        )
        self.stop_path = str(settings.get("stop_path") or "/sandboxes/{sandbox_id}/stop")
        self.health_path = str(settings.get("health_path") or "")
        self.health_check_interval_seconds = int(
            settings.get("health_check_interval_seconds") or 0
        )
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

    async def _ensure_sandbox_async(self, scope_key: str) -> str:
        now = time.monotonic()
        async with self._async_lock:
            await self._reap_idle_async(now)
            session = self._sessions.get(scope_key)
            if session is None:
                session = _SandboxSession(
                    handle=await self._create_sandbox_async(scope_key),
                    last_used=now,
                    last_checked=now,
                )
                self._sessions[scope_key] = session
            elif await self._should_check_health_async(session, now):
                if await self._sandbox_alive_async(session.handle):
                    session.last_checked = now
                else:
                    session = _SandboxSession(
                        handle=await self._create_sandbox_async(scope_key),
                        last_used=now,
                        last_checked=now,
                    )
                    self._sessions[scope_key] = session
            session.last_used = now
            return str(session.handle)

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
        payload = _compact_dict({
            "templateName": self.template_name,
            "sandboxId": _scoped_sandbox_id(self.settings, scope_key),
            "ossMountConfig": _skill_oss_mount_config(),
        })
        body = await self._request_async("POST", self.create_path, json=payload)
        sandbox_id = _extract_sandbox_id(body)
        if not sandbox_id:
            raise RuntimeError("sandbox REST create response missing sandboxId")
        logger.info(
            "sandbox REST created sandboxId={} scope={} skill_mounts={}",
            sandbox_id,
            scope_key,
            len(get_current_tool_scope().skill_mounts),
        )
        return str(sandbox_id)

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

    async def _stop_sandbox_async(self, handle) -> None:
        await self._request_async("POST", self.stop_path.format(sandbox_id=handle), json=None)

    async def _request_async(self, method: str, path: str, *, json: Optional[Dict[str, Any]]):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.api_key_header] = self.api_key
        if self.parent_id:
            headers["X-Acs-Parent-Id"] = self.parent_id
        url = f"{self.endpoint}{path if path.startswith('/') else '/' + path}"
        logger.info(
            "sandbox REST request {} {} headers={}",
            method,
            url,
            _mask_headers(headers),
        )
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.request(method, url, headers=headers, json=json)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "sandbox REST request failed status={} method={} url={} headers={} body={}",
                    exc.response.status_code,
                    method,
                    url,
                    _mask_headers(headers),
                    _response_text(exc.response),
                )
                if exc.response.status_code in {404, 410}:
                    raise SandboxUnavailable("cached sandbox is no longer available") from exc
                raise
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


def _mask_body(value: Any) -> Any:
    if isinstance(value, dict):
        masked = {}
        for key, item in value.items():
            if key in {"apiKey", "api_key", "accessKeySecret", "securityToken"}:
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


def _skill_oss_mount_config() -> Optional[Dict[str, Any]]:
    mount_points = [
        mount.get("oss")
        for mount in get_current_tool_scope().skill_mounts
        if isinstance(mount, dict) and mount.get("oss")
    ]
    if not mount_points:
        return None
    return {"mountPoints": mount_points}


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
        if not (
            settings.get("endpoint")
            and settings.get("template_name")
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
