"""Aliyun cross-account STS + PAI access, driven by the installed ``aliyun`` CLI.

This is the only module that shells out to ``aliyun``. It powers the "授权"
(authorize) flow: a customer creates a RAM role that trusts our developer
account with a per-user ExternalId; we ``sts:AssumeRole`` to mint ~1h temporary
credentials and verify the role can actually read the customer's PAI (EAS).

Command shapes here are the ones validated against the live Aliyun API:
  - ``aliyun sts AssumeRole --RoleArn ... --ExternalId ...``
  - ``aliyun sts GetCallerIdentity --region <r>``
  - ``aliyun PaiEas GET /api/v2/services --version 2021-07-01
      --endpoint pai-eas.<r>.aliyuncs.com --header "Content-Type=application/json" --force``

Credentials are NOT passed via env or ``--access-key-*`` flags: the installed CLI
(v3) ignores ``ALIBABACLOUD_*`` env credentials in ``--mode AK``/``StsToken`` and
falls back to the machine's ambient ``~/.aliyun/config.json`` ``current`` profile,
and argv is world-readable via procfs. Instead each call runs against a throwaway
``--config-path`` config (0600, deleted immediately) holding a single profile —
see ``_cli_config``. Nothing here logs raw credentials; callers get structured
dataclasses and keep the ``security_token`` out of logs/responses.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# acs:ram::<16-digit account id>:role/<name>. The character class also keeps
# shell/argv metacharacters out even though we never use shell=True.
_ARN_RE = re.compile(r"^acs:ram::\d{16}:role/[A-Za-z0-9+=,.@_-]{1,64}$")


class AliyunCliError(RuntimeError):
    """The ``aliyun`` binary is missing or a call failed unexpectedly."""


@dataclass
class Credentials:
    access_key_id: str
    access_key_secret: str
    security_token: str
    expiration: Optional[str] = None       # ISO8601 from AssumeRole
    assumed_role_arn: Optional[str] = None  # AssumedRoleUser.Arn


@dataclass
class Verdict:
    ok: bool
    stage: str  # "assume" | "identity" | "pai" | "creds"
    account_id: Optional[str] = None
    caller_arn: Optional[str] = None
    pai_total: Optional[int] = None  # aggregate across probed regions
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    # Per-region discovery snapshot: [{region, ok, pai_total, error_code}]. The
    # STS session credentials are global, so we probe several regions and report
    # where the customer actually has PAI/EAS services rather than pinning one.
    regions: Optional[List[dict]] = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "account_id": self.account_id,
            "caller_arn": self.caller_arn,
            "pai_total": self.pai_total,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "regions": self.regions,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def valid_role_arn(role_arn: str) -> bool:
    return bool(role_arn) and bool(_ARN_RE.match(role_arn))


def derive_external_id(user_id: str, *, secret: str, prefix: str = "pai-") -> str:
    """Stable, stateless per-user ExternalId.

    ``HMAC_SHA256(secret, user_id)`` truncated to 32 hex chars. Deterministic so
    it needs no storage, and the role's trust policy binds to exactly this value
    (confused-deputy defense). Fails closed when the secret is unset.
    """
    if not secret:
        raise ValueError("aliyun_authz_secret is not configured (ExternalId cannot be derived)")
    if not user_id:
        raise ValueError("user_id is required to derive an ExternalId")
    mac = hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).hexdigest()
    return f"{prefix}{mac[:32]}"


# RAM RoleName: letters/digits/hyphens, ≤64 chars. `pai-agent-<slug>-<12hex>`
# fits well within that and is safe to interpolate into the ROS URL / template.
_ROLE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_EMAIL_SLUG_CAP = 24


def _email_slug(email: Optional[str], cap: int = _EMAIL_SLUG_CAP) -> str:
    """Human-readable, RAM-safe label from an email's local part.

    ``Alice.Wong+test@x.com`` → ``alice-wong-test``. Lowercased, every run of
    non-alphanumerics collapsed to a single hyphen, trimmed and capped. Returns
    ``""`` when there's nothing usable (the hash alone then identifies the role).
    """
    local = (email or "").split("@", 1)[0].lower()
    slug = re.sub(r"[^a-z0-9]+", "-", local).strip("-")
    return slug[:cap].strip("-")


def derive_role_name(user_id: str, *, secret: str, email: Optional[str] = None,
                     prefix: str = "pai-agent-") -> str:
    """Stable, stateless per-user RAM role name.

    RAM roles are primary-account-scoped, so several app users binding the same
    Aliyun account (e.g. different RAM sub-users of one primary account) would
    collide on a single fixed name and clobber each other's trust policy. The
    name is ``<prefix><email-slug>-<12 hex>`` where the hex is
    ``HMAC_SHA256(secret, "role:"+user_id)`` — the slug makes the role legible in
    the RAM console, the hash guarantees uniqueness (email local parts aren't
    unique; the real owner key is user_id) and keeps it deterministic so
    re-authorizing updates the same role (idempotent ROS re-deploy). No email →
    hash only. Fails closed when the secret is unset."""
    if not secret:
        raise ValueError("aliyun_authz_secret is not configured (role name cannot be derived)")
    if not user_id:
        raise ValueError("user_id is required to derive a role name")
    mac = hmac.new(secret.encode(), f"role:{user_id}".encode(), hashlib.sha256).hexdigest()[:12]
    slug = _email_slug(email)
    body = f"{slug}-{mac}" if slug else mac
    return f"{prefix}{body}"


def valid_role_name(name: str) -> bool:
    return bool(name) and bool(_ROLE_NAME_RE.match(name))


# Developer long-term AK/SK are reused from the sandbox provider's env by
# default; the exact names are overridable via the aliyun_pai provider settings.
DEFAULT_BASE_AK_ENV = "AGENTRUN_ACCESS_KEY_ID"
DEFAULT_BASE_SK_ENV = "AGENTRUN_ACCESS_KEY_SECRET"


def provider_settings(agent_config) -> dict:
    """The ``aliyun_pai.default`` provider settings block, or ``{}`` (duck-typed
    so callers don't depend on the pydantic model)."""
    for p in getattr(agent_config, "providers", []) or []:
        if getattr(p, "id", "") == "aliyun_pai.default":
            return dict(getattr(p, "settings", {}) or {})
    return {}


def base_cred_env_names(settings_block: Optional[dict]) -> Tuple[str, str]:
    s = settings_block or {}
    return (
        str(s.get("base_access_key_id_env") or DEFAULT_BASE_AK_ENV),
        str(s.get("base_access_key_secret_env") or DEFAULT_BASE_SK_ENV),
    )


def read_base_creds(settings_block: Optional[dict], environ: Optional[dict] = None) -> Tuple[str, str]:
    """Read the developer base AK/SK from the env-var *names* declared in the
    aliyun_pai provider settings. Single source shared by the authorize route,
    the sandbox-env resolver, and the runtime-status check, so a custom env name
    can't make the UI report ``ready`` while the runtime silently reads nothing."""
    env = environ if environ is not None else os.environ
    ak_name, sk_name = base_cred_env_names(settings_block)
    return (env.get(ak_name, ""), env.get(sk_name, ""))


def configured_region(settings_block: Optional[dict], default_region: str) -> str:
    return str((settings_block or {}).get("region") or default_region or "cn-hangzhou")


# PAI/EAS-serving regions probed at authorize time when the provider does not
# declare an explicit `regions` list. Kept to a bounded, common set so the probe
# stays a handful of calls rather than sweeping every Aliyun region.
DEFAULT_PAI_REGIONS = [
    "cn-hangzhou", "cn-shanghai", "cn-beijing", "cn-shenzhen",
    "cn-hongkong", "ap-southeast-1",
]


def configured_regions(settings_block: Optional[dict], default_region: str) -> List[str]:
    """Ordered, de-duplicated candidate regions to probe for PAI services.

    An explicit ``regions: [...]`` in the provider settings wins verbatim;
    otherwise the configured/default single region is pinned first, followed by
    the common PAI region set. STS credentials are region-agnostic, so this list
    only governs *where we look* for services, never where a token is valid."""
    s = settings_block or {}
    raw = s.get("regions")
    if isinstance(raw, (list, tuple)) and raw:
        candidates = [str(r).strip() for r in raw if str(r).strip()]
    else:
        primary = configured_region(settings_block, default_region)
        candidates = [primary, *DEFAULT_PAI_REGIONS]
    seen: set = set()
    ordered: List[str] = []
    for r in candidates:
        if r and r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered or ["cn-hangzhou"]


def configured_ros_template_url(settings_block: Optional[dict], default_url: str) -> str:
    return str((settings_block or {}).get("ros_template_url") or default_url or "")


# The ROS template ships as a static file with `${DEVELOPER_ACCOUNT_ID}` /
# `${EXTERNAL_ID}` / `${ROLE_NAME}` placeholders so neither the trust-anchor
# account nor the per-user identifiers are hardcoded; they are injected when the
# template is served. parents[2] == the backend/ package root.
ROS_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "deploy" / "aliyun" / "authorize-role.yaml"
_ROS_ACCOUNT_PLACEHOLDER = "${DEVELOPER_ACCOUNT_ID}"
_ROS_EXTERNAL_ID_PLACEHOLDER = "${EXTERNAL_ID}"
_ROS_ROLE_NAME_PLACEHOLDER = "${ROLE_NAME}"
# Fallback when the template is fetched without a per-user role name.
_ROS_DEFAULT_ROLE_NAME = "pai-agent-crossaccount-role"
# App-generated identifiers only; validate before inlining so they can't break
# out of the single-quoted YAML scalar they become.
_EXTERNAL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def render_ros_template(
    developer_account_id: str,
    *,
    external_id: Optional[str] = None,
    role_name: Optional[str] = None,
    path: Optional[Path] = None,
) -> str:
    """Read the ROS template and substitute the developer account id (trust
    anchor) plus, when provided, the per-user ExternalId/RoleName as the
    parameters' ``Default`` values so the ROS create form is pre-filled. Fails
    closed when the account id is unset (the customer's role must trust a concrete
    account, never the literal placeholder). Per-user values are baked into the
    Defaults rather than passed as console query params because the path-style ROS
    console ignores top-level parameter query params."""
    if not developer_account_id:
        raise ValueError("aliyun_developer_account_id is not configured")
    if external_id is not None and not _EXTERNAL_ID_RE.match(external_id):
        raise ValueError("invalid external_id")
    if role_name is not None and not _ROLE_NAME_RE.match(role_name):
        raise ValueError("invalid role_name")
    text = (path or ROS_TEMPLATE_PATH).read_text(encoding="utf-8")
    text = text.replace(_ROS_ACCOUNT_PLACEHOLDER, developer_account_id)
    text = text.replace(_ROS_EXTERNAL_ID_PLACEHOLDER, external_id or "")
    text = text.replace(_ROS_ROLE_NAME_PLACEHOLDER, role_name or _ROS_DEFAULT_ROLE_NAME)
    return text


def configured_external_id_prefix(settings_block: Optional[dict]) -> str:
    return str((settings_block or {}).get("external_id_prefix") or "pai-")


def configured_role_name_prefix(settings_block: Optional[dict]) -> str:
    return str((settings_block or {}).get("role_name_prefix") or "pai-agent-")


# STS AssumeRole DurationSeconds bounds. The config floor is the RAM minimum
# (900s); operators may pick a shorter token if they want. The ceiling is the
# RAM maximum a role's MaxSessionDuration can be raised to (12h). The sandbox
# re-injects fresh creds before expiry regardless, so a longer token just
# reduces how often that refresh runs. _ROLE_MAX_FLOOR is the value every RAM
# role is guaranteed to accept (role MaxSessionDuration minimum), used as the
# retry target when a role rejects a longer requested duration.
_MIN_ASSUME_DURATION = 900
_ROLE_MAX_FLOOR = 3600
_MAX_ASSUME_DURATION = 43200


def configured_assume_duration_seconds(settings_block: Optional[dict]) -> int:
    raw = (settings_block or {}).get("assume_duration_seconds") or _MAX_ASSUME_DURATION
    try:
        duration = int(raw)
    except (TypeError, ValueError):
        duration = _MAX_ASSUME_DURATION
    return max(_MIN_ASSUME_DURATION, min(duration, _MAX_ASSUME_DURATION))


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text or "")


def parse_cli_error(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Pull ``ErrorCode``/``Message`` out of the CLI's noisy stderr."""
    clean = _strip_ansi(text).strip()
    code = message = None
    for line in clean.splitlines():
        line = line.strip()
        if line.startswith("ErrorCode:"):
            code = line.split(":", 1)[1].strip()
        elif line.startswith("Message:"):
            message = line.split(":", 1)[1].strip()
    if not (code or message):
        message = clean or None
    return code, message


# Substrings (matched case-insensitively against "<code> <message>") that tell a
# broken-credentials / not-authenticated failure — the ONLY class where showing a
# re-authorization card helps — apart from an authorization/permission failure,
# where the binding is fine but the role's policy doesn't cover the action (so a
# re-auth card would only confuse). Anything else (region, missing resource,
# network) is left to normal error reporting.
_CREDENTIAL_ERROR_SIGNATURES = (
    "invalidsecuritytoken",       # ...Expired / ...MismatchWithAccessKey / ...Malformed
    "securitytoken.expired",
    "invalidaccesskeyid",         # ...NotFound / .Inactive
    "missingsecuritytoken",
    "signaturedoesnotmatch",
    "sts.token",                  # credentials-go "no sts token" style
    "no credential",
    "unauthorized operation.aksk",
    "aliyun configure",           # CLI's own "please configure" hint = no profile
    "unknown profile",
    # No profile file at all — the unbound state. The CLI writes ~/.aliyun/config.json
    # only once STS creds are injected, so its absence means "not yet authorized",
    # which is exactly the "去授权" case. Anchored on CLI-internal phrasing / the
    # hidden config path so a user file literally named config.json can't trip it.
    "load configure failed",
    "/.aliyun/config",
)
_PERMISSION_ERROR_SIGNATURES = (
    "nopermission",
    "forbidden",
    "not authorized to",
    "you are not authorized",
    "no permission",
)


def classify_cli_error(code: Optional[str], message: Optional[str]) -> str:
    """Bucket an aliyun CLI failure into ``"credential" | "permission" | "other"``.

    ``credential`` = the caller isn't authenticated (missing/expired/invalid STS
    creds) → re-authorization can fix it. ``permission`` = authenticated but the
    role lacks the action → re-auth won't help, explain instead. ``other`` =
    everything else (region/resource/network)."""
    blob = f"{code or ''} {message or ''}".lower()
    if any(sig in blob for sig in _PERMISSION_ERROR_SIGNATURES):
        return "permission"
    if any(sig in blob for sig in _CREDENTIAL_ERROR_SIGNATURES):
        return "credential"
    return "other"


def _run(cmd: List[str], env: dict, timeout: float) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except FileNotFoundError as exc:  # aliyun not installed
        raise AliyunCliError("`aliyun` CLI not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise AliyunCliError(f"`aliyun {cmd[1] if len(cmd) > 1 else ''}` timed out") from exc
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


# Credential env vars that could otherwise sneak into a CLI call. We strip all of
# them from the child so only the throwaway ``--config-path`` profile provides
# credentials (deterministic regardless of the developer AK/SK injected at the
# process level for the sandbox, or a stale ambient profile).
_CRED_ENV_VARS = (
    "ALIBABACLOUD_ACCESS_KEY_ID", "ALIBABACLOUD_ACCESS_KEY_SECRET", "ALIBABACLOUD_SECURITY_TOKEN",
    "ALIBABA_CLOUD_ACCESS_KEY_ID", "ALIBABA_CLOUD_ACCESS_KEY_SECRET", "ALIBABA_CLOUD_SECURITY_TOKEN",
    "ALICLOUD_ACCESS_KEY_ID", "ALICLOUD_ACCESS_KEY_SECRET", "ALICLOUD_ACCESS_KEY",
    "ALICLOUD_SECRET_KEY", "ALICLOUD_SECURITY_TOKEN",
    "ACCESS_KEY_ID", "ACCESS_KEY_SECRET", "SECURITY_TOKEN",
)


def _clean_env() -> dict:
    env = dict(os.environ)
    for key in _CRED_ENV_VARS:
        env.pop(key, None)
    return env


@contextlib.contextmanager
def _cli_config(profile: dict):
    """Yield ``["--config-path", <path>]`` for a throwaway aliyun config whose only
    profile (named ``authz``, marked ``current``) carries ``profile``. The file is
    created 0600 and removed as soon as the block exits, so long-term/STS secrets
    never reach argv or the shared ``~/.aliyun/config.json``."""
    fd, path = tempfile.mkstemp(prefix="aliyun-authz-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump({"current": "authz", "profiles": [dict(profile, name="authz")]}, fh)
        yield ["--config-path", path]
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def to_sandbox_env(creds: Credentials) -> dict:
    """The three env vars the sandbox's ``aliyun`` CLI reads for a session role."""
    return {
        "ALIBABACLOUD_ACCESS_KEY_ID": creds.access_key_id,
        "ALIBABACLOUD_ACCESS_KEY_SECRET": creds.access_key_secret,
        "ALIBABACLOUD_SECURITY_TOKEN": creds.security_token,
    }


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def assume_role(
    role_arn: str,
    external_id: str,
    *,
    region: str,
    base_ak: str,
    base_sk: str,
    account_id: Optional[str] = None,
    role_session_name: str = "pai-agent",
    duration_seconds: int = 3600,
    timeout: float = 20.0,
) -> Credentials:
    """Assume the customer's cross-account role → temporary credentials.

    Signs the base call with the developer account's long-term AK/SK via a
    throwaway ``--config-path`` AK profile (so the machine's ambient
    ``~/.aliyun/config.json`` ``current`` profile can't hijack the call, and the
    secret never reaches argv). Raises ``AliyunCliError`` (message includes the
    parsed Aliyun ErrorCode) on failure.
    """
    if not valid_role_arn(role_arn):
        raise ValueError(f"invalid RoleArn: {role_arn!r}")
    if not (base_ak and base_sk):
        raise ValueError("developer base AK/SK not configured for AssumeRole")

    profile = {"mode": "AK", "access_key_id": base_ak,
               "access_key_secret": base_sk, "region_id": region}

    def _assume(duration: int):
        with _cli_config(profile) as cfg:
            cmd = [
                "aliyun", "sts", "AssumeRole", *cfg,
                "--RoleArn", role_arn,
                "--RoleSessionName", role_session_name,
                "--ExternalId", external_id,
                "--DurationSeconds", str(duration),
                "--region", region,
            ]
            return _run(cmd, _clean_env(), timeout)

    rc, out, err = _assume(duration_seconds)
    if rc != 0:
        code, msg = parse_cli_error(err or out)
        # A role whose MaxSessionDuration is smaller than the requested
        # DurationSeconds rejects the call. Retry once at the STS floor (3600s)
        # so roles created before we raised MaxSessionDuration keep working
        # without forcing the customer to re-deploy the ROS stack.
        blob = f"{code} {msg}".lower()
        if duration_seconds > _ROLE_MAX_FLOOR and (
            "durationseconds" in blob
            or "maxsessionduration" in blob
            or "session duration" in blob
        ):
            rc, out, err = _assume(_ROLE_MAX_FLOOR)
            if rc != 0:
                code, msg = parse_cli_error(err or out)
        if rc != 0:
            raise AliyunCliError(f"AssumeRole failed: {code or 'error'}: {msg or 'unknown'}")
    try:
        doc = json.loads(out)
        c = doc["Credentials"]
        return Credentials(
            access_key_id=c["AccessKeyId"],
            access_key_secret=c["AccessKeySecret"],
            security_token=c["SecurityToken"],
            expiration=c.get("Expiration"),
            assumed_role_arn=(doc.get("AssumedRoleUser") or {}).get("Arn"),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AliyunCliError("AssumeRole returned an unparseable response") from exc


def _list_pai_services(cfg: List[str], env: dict, region: str,
                       timeout: float) -> Tuple[bool, Optional[int], Optional[str]]:
    """One region's read-only PAI/EAS list → (ok, total, error_code). ``ok`` means
    the endpoint answered (HTTP 200); ``total`` may still be None on odd bodies.
    ``cfg`` is the ``--config-path`` pair from the caller's ``_cli_config``."""
    rc, out, err = _run(
        ["aliyun", "PaiEas", "GET", "/api/v2/services", *cfg,
         "--version", "2021-07-01",
         "--endpoint", f"pai-eas.{region}.aliyuncs.com",
         "--header", "Content-Type=application/json",
         "--force"],
        env, timeout,
    )
    if rc != 0:
        code, _ = parse_cli_error(err or out)
        return False, None, code
    try:
        eas = json.loads(out)
        total = eas.get("TotalCount")
        if total is None:
            services = eas.get("Services") or eas.get("services") or []
            total = len(services) if isinstance(services, list) else 0
    except json.JSONDecodeError:
        total = None
    return True, total, None


def verify_pai_access(creds: Credentials, *, regions: List[str], timeout: float = 20.0) -> Verdict:
    """Identity check once, then a PAI/EAS list in each candidate region.

    STS session credentials are region-agnostic, so instead of pinning one region
    we probe several and record where services actually live. The verdict is
    ``ok`` when identity resolves AND at least one region's PAI endpoint answers
    (any service count, including zero) — a single region being disabled or empty
    no longer fails the whole authorization. Only if *every* region errors do we
    return ``ok=False`` (surfacing the first error). Never raises for a normal
    Aliyun error; only a missing CLI raises.
    """
    env = _clean_env()
    candidates = [r for r in (regions or []) if r] or ["cn-hangzhou"]
    profile = {"mode": "StsToken",
               "access_key_id": creds.access_key_id,
               "access_key_secret": creds.access_key_secret,
               "sts_token": creds.security_token,
               "region_id": candidates[0]}

    per_region: List[dict] = []
    agg_total = 0
    saw_total = False
    any_ok = False
    first_err: Tuple[Optional[str], Optional[str]] = (None, None)
    with _cli_config(profile) as cfg:
        rc, out, err = _run(
            ["aliyun", "sts", "GetCallerIdentity", *cfg, "--region", candidates[0]],
            env, timeout,
        )
        if rc != 0:
            code, msg = parse_cli_error(err or out)
            return Verdict(ok=False, stage="identity", error_code=code, error_message=msg)
        try:
            ident = json.loads(out)
        except json.JSONDecodeError:
            ident = {}
        account_id = ident.get("AccountId")
        caller_arn = ident.get("Arn")

        for region in candidates:
            ok, total, code = _list_pai_services(cfg, env, region, timeout)
            per_region.append({"region": region, "ok": ok, "pai_total": total, "error_code": code})
            if not ok:
                if first_err == (None, None):
                    first_err = (code, None)
                continue
            any_ok = True
            if isinstance(total, int):
                agg_total += total
                saw_total = True

    if not any_ok:
        code, msg = first_err
        return Verdict(ok=False, stage="pai", account_id=account_id, caller_arn=caller_arn,
                       error_code=code, error_message=msg, regions=per_region)

    return Verdict(ok=True, stage="pai", account_id=account_id, caller_arn=caller_arn,
                   pai_total=agg_total if saw_total else None, regions=per_region)
