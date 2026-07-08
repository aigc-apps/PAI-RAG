import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import types
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps import AppState
from app.routes.aliyun import router as aliyun_router
from app.store.memory import InMemoryStore
from agent.integrations import aliyun_sts
from tests.authutil import apply_auth


VALID_ARN = "acs:ram::1730760139076263:role/feiyue-test-role"


def _client(monkeypatch, *, secret="k3y", base_creds=True, ros=True,
            account_id="1095312831785714", agent_config=None):
    monkeypatch.setenv("ALIYUN_AUTHZ_SECRET", secret)
    monkeypatch.setenv("ALIYUN_DEFAULT_REGION", "cn-hangzhou")
    monkeypatch.setenv("ALIYUN_ROS_TEMPLATE_URL",
                       "https://ex.oss.com/authorize-role.yaml" if ros else "")
    if account_id:
        monkeypatch.setenv("ALIYUN_DEVELOPER_ACCOUNT_ID", account_id)
    else:
        monkeypatch.delenv("ALIYUN_DEVELOPER_ACCOUNT_ID", raising=False)
    if base_creds:
        monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "AK")
        monkeypatch.setenv("AGENTRUN_ACCESS_KEY_SECRET", "SK")
    else:
        monkeypatch.delenv("AGENTRUN_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AGENTRUN_ACCESS_KEY_SECRET", raising=False)
    store = InMemoryStore()
    app = FastAPI()
    app.state.app_state = AppState(
        store=store, llm=None, default_model="x", agent_config=agent_config)
    app.include_router(aliyun_router)
    # Identity is now the authenticated user; the tests bind/read as u1.
    return TestClient(apply_auth(app, user_id="u1", role="user")), store


def _ok_creds(*a, **k):
    return aliyun_sts.Credentials(access_key_id="STS.x", access_key_secret="s", security_token="TOKEN")


def _agent_config(settings):
    return types.SimpleNamespace(
        providers=[types.SimpleNamespace(id="aliyun_pai.default", settings=settings)])


def test_authorize_happy_path_persists_and_hides_creds(monkeypatch):
    c, store = _client(monkeypatch)
    monkeypatch.setattr(aliyun_sts, "assume_role", _ok_creds)
    monkeypatch.setattr(aliyun_sts, "verify_pai_access",
                        lambda creds, regions: aliyun_sts.Verdict(
                            ok=True, stage="pai", account_id="1730760139076263", pai_total=3))

    r = c.post("/v1/aliyun/authorize", json={"user_id": "u1", "role_arn": VALID_ARN})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["external_id"].startswith("pai-")
    assert body["verdict"]["account_id"] == "1730760139076263"
    # no credential material leaks into the response
    assert "TOKEN" not in r.text and "STS.x" not in r.text

    user = store._users["u1"]
    binding = user.meta["aliyun_pai"]
    assert binding["role_arn"] == VALID_ARN
    assert binding["assumed_account_id"] == "1730760139076263"
    assert "security_token" not in binding and "SecurityToken" not in str(binding)


def test_authorize_rejects_bad_arn(monkeypatch):
    c, _ = _client(monkeypatch)
    r = c.post("/v1/aliyun/authorize", json={"user_id": "u1", "role_arn": "bad;arn"})
    assert r.status_code == 400


def test_authorize_missing_base_creds_503(monkeypatch):
    c, _ = _client(monkeypatch, base_creds=False)
    r = c.post("/v1/aliyun/authorize", json={"user_id": "u1", "role_arn": VALID_ARN})
    assert r.status_code == 503


def test_authorize_missing_secret_503(monkeypatch):
    c, _ = _client(monkeypatch, secret="")
    r = c.post("/v1/aliyun/authorize", json={"user_id": "u1", "role_arn": VALID_ARN})
    assert r.status_code == 503


def test_authorize_sts_error_does_not_persist(monkeypatch):
    c, store = _client(monkeypatch)

    def boom(*a, **k):
        raise aliyun_sts.AliyunCliError("AssumeRole failed: NoPermission: nope")

    monkeypatch.setattr(aliyun_sts, "assume_role", boom)
    r = c.post("/v1/aliyun/authorize", json={"user_id": "u1", "role_arn": VALID_ARN})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert "u1" not in store._users or "aliyun_pai" not in (store._users["u1"].meta or {})


def test_authorize_verify_fail_does_not_persist(monkeypatch):
    c, store = _client(monkeypatch)
    monkeypatch.setattr(aliyun_sts, "assume_role", _ok_creds)
    monkeypatch.setattr(aliyun_sts, "verify_pai_access",
                        lambda creds, regions: aliyun_sts.Verdict(
                            ok=False, stage="pai", error_code="NoPermission"))
    r = c.post("/v1/aliyun/authorize", json={"user_id": "u1", "role_arn": VALID_ARN})
    assert r.json()["ok"] is False
    assert "u1" not in store._users or "aliyun_pai" not in (store._users["u1"].meta or {})


def test_status_unbound_then_bound(monkeypatch):
    c, store = _client(monkeypatch)
    r = c.get("/v1/aliyun/status", params={"user_id": "u1"}).json()
    assert r["bound"] is False
    assert r["external_id"].startswith("pai-")
    assert r["ros_url"] and "templateUrl=" in r["ros_url"] and "ExternalId=" in r["ros_url"]
    # Per-user role name is derived and threaded into the one-click link so
    # multiple users binding the same Aliyun account don't collide.
    assert r["role_name"] and r["role_name"].startswith("pai-agent-")
    assert f"RoleName={r['role_name']}" in r["ros_url"]
    assert r["configured"] is True

    monkeypatch.setattr(aliyun_sts, "assume_role", _ok_creds)
    monkeypatch.setattr(aliyun_sts, "verify_pai_access",
                        lambda creds, regions: aliyun_sts.Verdict(
                            ok=True, stage="pai", account_id="17", pai_total=0))
    c.post("/v1/aliyun/authorize", json={"user_id": "u1", "role_arn": VALID_ARN})
    r = c.get("/v1/aliyun/status", params={"user_id": "u1"}).json()
    assert r["bound"] is True and r["role_arn"] == VALID_ARN


def test_status_uses_provider_ros_url_when_env_url_missing(monkeypatch):
    c, _ = _client(
        monkeypatch,
        ros=False,
        agent_config=_agent_config({
            "ros_template_url": "https://provider.example.com/authorize-role.yaml",
        }),
    )

    r = c.get("/v1/aliyun/status", params={"user_id": "u1"}).json()

    assert r["configured"] is True
    assert "provider.example.com" in r["ros_url"]


def test_authorize_uses_provider_region_prefix_and_duration(monkeypatch):
    captured = {}
    c, _ = _client(
        monkeypatch,
        agent_config=_agent_config({
            "region": "cn-shanghai",
            "external_id_prefix": "pai-test-",
            "assume_duration_seconds": 1800,
        }),
    )

    def assume_role(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _ok_creds()

    monkeypatch.setattr(aliyun_sts, "assume_role", assume_role)
    monkeypatch.setattr(aliyun_sts, "verify_pai_access",
                        lambda creds, regions: aliyun_sts.Verdict(
                            ok=True, stage="pai", account_id="17", pai_total=0))

    r = c.post("/v1/aliyun/authorize", json={"user_id": "u1", "role_arn": VALID_ARN})

    assert r.status_code == 200
    assert r.json()["external_id"].startswith("pai-test-")
    assert captured["kwargs"]["region"] == "cn-shanghai"
    assert captured["kwargs"]["duration_seconds"] == 1800


def test_deauthorize_clears_binding(monkeypatch):
    c, store = _client(monkeypatch)
    monkeypatch.setattr(aliyun_sts, "assume_role", _ok_creds)
    monkeypatch.setattr(aliyun_sts, "verify_pai_access",
                        lambda creds, regions: aliyun_sts.Verdict(ok=True, stage="pai", account_id="17"))
    c.post("/v1/aliyun/authorize", json={"user_id": "u1", "role_arn": VALID_ARN})
    r = c.post("/v1/aliyun/deauthorize", json={"user_id": "u1"})
    assert r.status_code == 200 and r.json()["bound"] is False
    assert store._users["u1"].meta["aliyun_pai"] is None


def _seed_binding(store, *, role_arn=VALID_ARN, external_id="pai-seed",
                  default_region="cn-hangzhou"):
    from app.store.base import User
    store._users["u1"] = User(id="u1", email="test@example.com", meta={"aliyun_pai": {
        "role_arn": role_arn, "external_id": external_id,
        "regions": ["cn-hangzhou"], "service_regions": [],
        "default_region": default_region, "last_verdict": "ok",
    }})


def test_verify_404_without_binding(monkeypatch):
    c, _ = _client(monkeypatch)
    r = c.post("/v1/aliyun/verify", json={"user_id": "u1"})
    assert r.status_code == 404


def test_verify_refreshes_binding_on_success(monkeypatch):
    c, store = _client(monkeypatch)
    _seed_binding(store, external_id="pai-old", default_region="cn-hangzhou")
    captured = {}

    def assume_role(role_arn, external_id, **kwargs):
        captured["role_arn"] = role_arn
        captured["external_id"] = external_id
        return _ok_creds()

    monkeypatch.setattr(aliyun_sts, "assume_role", assume_role)
    monkeypatch.setattr(aliyun_sts, "verify_pai_access",
                        lambda creds, regions: aliyun_sts.Verdict(
                            ok=True, stage="pai", account_id="17", pai_total=5,
                            regions=[{"region": "cn-shanghai", "ok": True, "pai_total": 5}]))

    r = c.post("/v1/aliyun/verify", json={"user_id": "u1"})
    assert r.status_code == 200 and r.json()["ok"] is True
    # Re-verify reuses the STORED role_arn/external_id (no re-authorize).
    assert captured["role_arn"] == VALID_ARN
    assert captured["external_id"] == "pai-old"
    binding = store._users["u1"].meta["aliyun_pai"]
    assert binding["default_region"] == "cn-shanghai"
    assert binding["service_regions"] == ["cn-shanghai"]
    assert "TOKEN" not in r.text and "STS.x" not in r.text


def test_verify_failure_keeps_binding(monkeypatch):
    c, store = _client(monkeypatch)
    _seed_binding(store)

    def boom(*a, **k):
        raise aliyun_sts.AliyunCliError("AssumeRole failed: InvalidSecurityToken.Expired")

    monkeypatch.setattr(aliyun_sts, "assume_role", boom)
    r = c.post("/v1/aliyun/verify", json={"user_id": "u1"})
    assert r.status_code == 200 and r.json()["ok"] is False
    # A transient failure must NOT drop a good binding.
    assert store._users["u1"].meta["aliyun_pai"]["role_arn"] == VALID_ARN


def test_ros_template_injects_configured_account_id(monkeypatch):
    c, _ = _client(monkeypatch, account_id="1122334455667788")
    r = c.get("/v1/aliyun/ros-template.yaml")
    assert r.status_code == 200
    assert "acs:ram::1122334455667788:root" in r.text
    # no hardcoded id and no leftover placeholder
    assert "1095312831785714" not in r.text
    assert "${DEVELOPER_ACCOUNT_ID}" not in r.text


def test_ros_template_503_without_account_id(monkeypatch):
    c, _ = _client(monkeypatch, account_id="")
    assert c.get("/v1/aliyun/ros-template.yaml").status_code == 503


def test_ros_template_bakes_per_user_defaults_from_query(monkeypatch):
    # The ROS console fetches this via the link's templateUrl (which carries the
    # per-user query); the values must land as the parameters' Defaults so the
    # create form is pre-filled.
    c, _ = _client(monkeypatch, account_id="1122334455667788")
    r = c.get("/v1/aliyun/ros-template.yaml", params={
        "external_id": "pai-159a4f64385292c3cafa867954672314",
        "role_name": "pai-agent-feiyue-c7ab388b5613",
    })
    assert r.status_code == 200
    assert "Default: 'pai-159a4f64385292c3cafa867954672314'" in r.text
    assert "Default: 'pai-agent-feiyue-c7ab388b5613'" in r.text


def test_status_self_hosts_template_when_no_url(monkeypatch):
    # No published URL, but the developer account id is set → the app self-hosts
    # the rendered template and reports configured=True with a working ros_url.
    c, _ = _client(monkeypatch, ros=False, account_id="1122334455667788")
    r = c.get("/v1/aliyun/status", params={"user_id": "u1"}).json()
    assert r["configured"] is True
    assert "ros-template.yaml" in r["ros_url"] and "ExternalId=" in r["ros_url"]
    # The self-hosted templateUrl carries the per-user query so the endpoint can
    # bake ExternalId/RoleName into the rendered template's Defaults.
    assert "external_id" in r["ros_url"] and "role_name" in r["ros_url"]


def test_status_self_hosted_template_uses_public_base_url(monkeypatch):
    # PUBLIC_BASE_URL overrides the request host so the ROS console fetches the
    # template from the canonical public origin, not whatever proxy/frontend host
    # the request arrived on.
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://pai.example.com/")
    c, _ = _client(monkeypatch, ros=False, account_id="1122334455667788")
    r = c.get("/v1/aliyun/status", params={"user_id": "u1"}).json()
    assert "https%3A%2F%2Fpai.example.com%2Fv1%2Faliyun%2Fros-template.yaml" in r["ros_url"]
    assert "testserver" not in r["ros_url"]


def test_status_not_configured_without_url_or_account_id(monkeypatch):
    c, _ = _client(monkeypatch, ros=False, account_id="")
    r = c.get("/v1/aliyun/status", params={"user_id": "u1"}).json()
    assert r["configured"] is False
    assert r["ros_url"] is None
