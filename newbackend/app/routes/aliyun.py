"""Per-user Aliyun PAI cross-account authorization.

Self-service (keyed on the caller's ``user_id``, like the responses route — not
the admin config gate). The customer creates a RAM role via the ROS one-click
link, pastes the RoleArn back, and we AssumeRole + verify PAI access before
persisting the binding. Only the durable ``role_arn`` + ``external_id`` are
stored; temporary credentials are never returned or persisted.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from urllib.parse import quote, urlencode
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from pydantic import BaseModel

from agent.integrations import aliyun_sts
from app.auth import require_user
from app.config import get_settings
from app.deps import AppState, get_state
from app.store.base import User
from app.urls import external_base_url

router = APIRouter(prefix="/v1/aliyun", tags=["aliyun"])


class AuthorizePayload(BaseModel):
    role_arn: str


def _base_creds(agent_config) -> tuple[str, str]:
    """Developer account long-term AK/SK, read from the env-var names declared in
    the aliyun_pai provider settings (default AGENTRUN_ACCESS_KEY_ID/_SECRET)."""
    return aliyun_sts.read_base_creds(aliyun_sts.provider_settings(agent_config))


def _provider_settings(agent_config) -> dict:
    return aliyun_sts.provider_settings(agent_config)


def _region(settings, provider_settings: dict) -> str:
    return aliyun_sts.configured_region(provider_settings, settings.aliyun_default_region)


def _regions(settings, provider_settings: dict) -> list[str]:
    """Candidate regions to probe/advertise. STS creds are global; this only
    governs where we look for the customer's PAI services."""
    return aliyun_sts.configured_regions(provider_settings, settings.aliyun_default_region)


def _self_hosted_template_url(request: Optional[Request]) -> Optional[str]:
    """Absolute URL of the app's own rendered ROS template endpoint. Uses the
    configured PUBLIC_BASE_URL when set, else the incoming request's base_url. Only
    usable when the deployment is publicly reachable by the Aliyun ROS console — so
    set PUBLIC_BASE_URL to the public origin when requests arrive via a proxy that
    rewrites Host to an internal/frontend address."""
    base = external_base_url(request)
    return f"{base}/v1/aliyun/ros-template.yaml" if base else None


def _resolve_template_url(settings, provider_settings: dict,
                          request: Optional[Request] = None, *,
                          external_id: Optional[str] = None,
                          role_name: Optional[str] = None) -> str:
    """Explicit ALIYUN_ROS_TEMPLATE_URL / provider url wins; otherwise fall back to
    the self-hosted rendered template (available once the developer account id is
    configured, so the account is never hardcoded into a published file).

    For the self-hosted template we bake the per-user ExternalId/RoleName into the
    URL's query, so the endpoint renders them as the parameters' Defaults and the
    ROS create form is pre-filled (the path-style console ignores its own top-level
    parameter query params). An explicit/published URL is static and can't be
    rendered per-user, so it's returned unchanged."""
    explicit = aliyun_sts.configured_ros_template_url(
        provider_settings, settings.aliyun_ros_template_url)
    if explicit:
        return explicit
    if settings.aliyun_developer_account_id:
        base = _self_hosted_template_url(request)
        if not base:
            return ""
        params = {}
        if external_id:
            params["external_id"] = external_id
        if role_name:
            params["role_name"] = role_name
        return base + ("?" + urlencode(params) if params else "")
    return ""


def _ros_url(template_url: str, external_id: str, region: str,
             role_name: Optional[str] = None) -> Optional[str]:
    if not template_url:
        return None
    url = (
        f"https://ros.console.aliyun.com/{region}/stacks/create"
        f"?templateUrl={quote(template_url, safe='')}"
        f"&ExternalId={quote(external_id, safe='')}"
    )
    # Per-user role name so multiple users binding the same Aliyun account don't
    # collide on one fixed RoleName (RAM roles are primary-account-scoped).
    if role_name:
        url += f"&RoleName={quote(role_name, safe='')}"
    # NOTE: the path-style ROS console ignores these top-level ExternalId/RoleName
    # query params — the reliable prefill is the per-user Default baked into the
    # self-hosted template (see _resolve_template_url / render_ros_template). These
    # are kept as best-effort for any console view that does honor them.
    return url


def _binding_from_verdict(role_arn: str, external_id: str, verdict, regions: list) -> dict:
    """Shape a successful verdict into the stored ``aliyun_pai`` binding. No
    credential material — only where services were discovered so the sandbox/UI
    know which regions to work in."""
    probed = verdict.regions or []
    reachable = [r["region"] for r in probed if r.get("ok")]
    service_regions = [r["region"] for r in probed if r.get("ok") and (r.get("pai_total") or 0) > 0]
    region_totals = {
        r["region"]: r.get("pai_total")
        for r in probed if r.get("ok") and r.get("pai_total") is not None
    }
    default_region = (service_regions or reachable or regions)[0]
    return {
        "role_arn": role_arn,
        "external_id": external_id,
        "regions": reachable,
        "service_regions": service_regions,
        "region_totals": region_totals,
        "default_region": default_region,
        "assumed_account_id": verdict.account_id,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "last_verdict": "ok",
    }


@router.post("/authorize")
async def authorize(payload: AuthorizePayload, state: AppState = Depends(get_state),
                    user: User = Depends(require_user)):
    settings = get_settings()
    pai_settings = _provider_settings(state.agent_config)
    regions = _regions(settings, pai_settings)

    if not aliyun_sts.valid_role_arn(payload.role_arn):
        raise HTTPException(status_code=400, detail="invalid RoleArn format")
    try:
        external_id = aliyun_sts.derive_external_id(
            user.id,
            secret=settings.aliyun_authz_secret,
            prefix=aliyun_sts.configured_external_id_prefix(pai_settings),
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    base_ak, base_sk = _base_creds(state.agent_config)
    if not (base_ak and base_sk):
        raise HTTPException(status_code=503,
                            detail="developer base credentials not configured")

    # AssumeRole + verify (blocking subprocess → thread). A normal Aliyun failure
    # returns a verdict; only infra errors (missing CLI, bad ARN) raise.
    try:
        creds = await asyncio.to_thread(
            aliyun_sts.assume_role, payload.role_arn, external_id,
            region=regions[0], base_ak=base_ak, base_sk=base_sk,
            account_id=settings.aliyun_developer_account_id,
            duration_seconds=aliyun_sts.configured_assume_duration_seconds(pai_settings),
        )
    except aliyun_sts.AliyunCliError as exc:
        logger.info("aliyun authorize: AssumeRole failed for user={}: {}", user.id, exc)
        return JSONResponse({
            "ok": False,
            "external_id": external_id,
            "verdict": {"ok": False, "stage": "assume", "error_message": str(exc)},
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    verdict = await asyncio.to_thread(aliyun_sts.verify_pai_access, creds, regions=regions)

    if verdict.ok:
        binding = _binding_from_verdict(payload.role_arn, external_id, verdict, regions)
        await state.store.update_user_meta(user.id, {"aliyun_pai": binding})
        logger.info("aliyun authorize: user={} bound to account={} (services in {})",
                    user.id, verdict.account_id, binding["service_regions"] or "none discovered")

    return JSONResponse({
        "ok": verdict.ok,
        "external_id": external_id,
        "verdict": verdict.as_dict(),
    })


@router.get("/ros-template.yaml")
async def ros_template(external_id: Optional[str] = None,
                       role_name: Optional[str] = None):
    """Public: the ROS template with the developer account id injected from config
    and the per-user ExternalId/RoleName baked in as the parameters' Defaults so
    the create form is pre-filled. The Aliyun ROS console fetches this anonymously
    via the one-click link (query params carried in the link's templateUrl), so it
    carries no auth. The account id is never hardcoded — 503 until it is set. The
    per-user identifiers are not credentials: a mismatched ExternalId simply fails
    to AssumeRole at runtime, so accepting them from the query is safe."""
    settings = get_settings()
    try:
        body = aliyun_sts.render_ros_template(
            settings.aliyun_developer_account_id,
            external_id=external_id,
            role_name=role_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except OSError as exc:  # template file missing/unreadable
        raise HTTPException(status_code=500, detail=f"ROS template unavailable: {exc}")
    return PlainTextResponse(body, media_type="application/x-yaml")


@router.get("/status")
async def status(request: Request, state: AppState = Depends(get_state),
                 user: User = Depends(require_user)):
    settings = get_settings()
    pai_settings = _provider_settings(state.agent_config)
    # Primary region only drives the region-scoped ROS console link; the binding
    # itself carries the full set of regions where services were discovered.
    region = _region(settings, pai_settings)
    external_id = None
    role_name = None
    try:
        external_id = aliyun_sts.derive_external_id(
            user.id,
            secret=settings.aliyun_authz_secret,
            prefix=aliyun_sts.configured_external_id_prefix(pai_settings),
        )
        role_name = aliyun_sts.derive_role_name(
            user.id,
            secret=settings.aliyun_authz_secret,
            email=user.email,
            prefix=aliyun_sts.configured_role_name_prefix(pai_settings),
        )
    except ValueError:
        pass  # secret unset → feature not configured; still report unbound

    # Resolve the template URL after deriving the identifiers so the self-hosted
    # template carries them (baked into its parameter Defaults on render).
    template_url = _resolve_template_url(
        settings, pai_settings, request,
        external_id=external_id, role_name=role_name,
    )

    stored = await state.store.get_user(user.id)
    binding = (stored.meta or {}).get("aliyun_pai") if stored else None

    # "configured" mirrors the exact runtime prerequisites so the UI can't show a
    # ready state while authorize would 503: HMAC secret + a resolvable ROS
    # template (explicit URL, or self-hosted once the developer account id is set)
    # + the developer base AK/SK present under their declared env names.
    base_ak, base_sk = _base_creds(state.agent_config)
    configured = bool(
        settings.aliyun_authz_secret
        and template_url
        and base_ak and base_sk
    )

    return JSONResponse({
        "bound": bool(binding),
        "region": region,
        # Multi-region discovery from the last authorization (empty when unbound
        # or from a legacy single-region binding).
        "regions": (binding.get("regions") if binding else None) or [],
        "service_regions": (binding.get("service_regions") if binding else None) or [],
        "region_totals": (binding.get("region_totals") if binding else None) or {},
        "default_region": binding.get("default_region") if binding else None,
        "external_id": external_id,
        "role_arn": binding.get("role_arn") if binding else None,
        "assumed_account_id": binding.get("assumed_account_id") if binding else None,
        "verified_at": binding.get("verified_at") if binding else None,
        "ros_url": _ros_url(template_url, external_id, region, role_name) if external_id else None,
        "role_name": role_name,
        "configured": configured,
    })


@router.post("/verify")
async def verify(state: AppState = Depends(get_state),
                 user: User = Depends(require_user)):
    """Re-run AssumeRole + PAI probing against the user's EXISTING binding — no
    re-authorization. Health-check for the "authorized but CLI still errors"
    case; refreshes the binding's region snapshot on success, leaves it intact on
    failure (so a transient outage doesn't drop a good binding)."""
    settings = get_settings()
    pai_settings = _provider_settings(state.agent_config)
    regions = _regions(settings, pai_settings)

    stored = await state.store.get_user(user.id)
    binding = (stored.meta or {}).get("aliyun_pai") if stored else None
    if not binding or not binding.get("role_arn"):
        raise HTTPException(status_code=404, detail="no aliyun binding to verify")

    role_arn = binding["role_arn"]
    try:
        external_id = binding.get("external_id") or aliyun_sts.derive_external_id(
            user.id,
            secret=settings.aliyun_authz_secret,
            prefix=aliyun_sts.configured_external_id_prefix(pai_settings),
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    base_ak, base_sk = _base_creds(state.agent_config)
    if not (base_ak and base_sk):
        raise HTTPException(status_code=503,
                            detail="developer base credentials not configured")

    try:
        creds = await asyncio.to_thread(
            aliyun_sts.assume_role, role_arn, external_id,
            region=regions[0], base_ak=base_ak, base_sk=base_sk,
            account_id=settings.aliyun_developer_account_id,
            duration_seconds=aliyun_sts.configured_assume_duration_seconds(pai_settings),
        )
    except aliyun_sts.AliyunCliError as exc:
        logger.info("aliyun verify: AssumeRole failed for user={}: {}", user.id, exc)
        return JSONResponse({
            "ok": False,
            "external_id": external_id,
            "verdict": {"ok": False, "stage": "assume", "error_message": str(exc)},
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    verdict = await asyncio.to_thread(aliyun_sts.verify_pai_access, creds, regions=regions)
    if verdict.ok:
        refreshed = _binding_from_verdict(role_arn, external_id, verdict, regions)
        await state.store.update_user_meta(user.id, {"aliyun_pai": refreshed})

    return JSONResponse({
        "ok": verdict.ok,
        "external_id": external_id,
        "verdict": verdict.as_dict(),
    })


@router.post("/deauthorize")
async def deauthorize(state: AppState = Depends(get_state),
                      user: User = Depends(require_user)):
    await state.store.update_user_meta(user.id, {"aliyun_pai": None})
    return JSONResponse({"ok": True, "bound": False})
