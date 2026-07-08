import sys, os, asyncio, types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.builder import _resolve_aliyun_sandbox_env
from app.store.memory import InMemoryStore
from agent.integrations import aliyun_sts
from agent.tools.scope import ToolScope
from agent.tools.sandbox_providers import _build_env_contract, _mask_body


def _agent_config(enabled=True):
    return types.SimpleNamespace(
        capabilities=[types.SimpleNamespace(id="aliyun_pai", enabled=enabled)])


def _env(monkeypatch, secret="k3y", base=True):
    monkeypatch.setenv("ALIYUN_AUTHZ_SECRET", secret)
    monkeypatch.setenv("ALIYUN_DEFAULT_REGION", "cn-hangzhou")
    if base:
        monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "AK")
        monkeypatch.setenv("AGENTRUN_ACCESS_KEY_SECRET", "SK")
    else:
        monkeypatch.delenv("AGENTRUN_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AGENTRUN_ACCESS_KEY_SECRET", raising=False)


async def _store_with_binding():
    store = InMemoryStore()
    await store.ensure_user("u1")
    await store.update_user_meta("u1", {"aliyun_pai": {
        "role_arn": "acs:ram::1730760139076263:role/r",
        "external_id": "pai-abc", "region": "cn-hangzhou"}})
    return store


def test_resolver_injects_sts_keys_and_region_when_bound(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(aliyun_sts, "assume_role", lambda *a, **k: aliyun_sts.Credentials(
        access_key_id="STS.x", access_key_secret="s", security_token="TOKEN"))

    async def run():
        store = await _store_with_binding()
        return await _resolve_aliyun_sandbox_env(store, _agent_config(), "u1")

    env = asyncio.run(run())
    # Three session-credential keys plus a default region hint. The legacy binding
    # carries only `region`, so no PAI_AVAILABLE_REGIONS list is emitted.
    assert env == {
        "ALIBABACLOUD_ACCESS_KEY_ID": "STS.x",
        "ALIBABACLOUD_ACCESS_KEY_SECRET": "s",
        "ALIBABACLOUD_SECURITY_TOKEN": "TOKEN",
        "ALIBABACLOUD_REGION_ID": "cn-hangzhou",
    }


def test_resolver_emits_available_regions_from_binding(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(aliyun_sts, "assume_role", lambda *a, **k: aliyun_sts.Credentials(
        access_key_id="STS.x", access_key_secret="s", security_token="TOKEN"))

    async def run():
        store = InMemoryStore()
        await store.ensure_user("u1")
        await store.update_user_meta("u1", {"aliyun_pai": {
            "role_arn": "acs:ram::1730760139076263:role/r",
            "external_id": "pai-abc",
            "regions": ["cn-shanghai", "cn-beijing"],
            "service_regions": ["cn-shanghai"],
            "default_region": "cn-shanghai",
        }})
        return await _resolve_aliyun_sandbox_env(store, _agent_config(), "u1")

    env = asyncio.run(run())
    assert env["ALIBABACLOUD_REGION_ID"] == "cn-shanghai"
    assert env["PAI_AVAILABLE_REGIONS"] == "cn-shanghai"


def test_resolver_skips_when_capability_disabled(monkeypatch):
    _env(monkeypatch)
    monkeypatch.setattr(aliyun_sts, "assume_role", lambda *a, **k: 1 / 0)  # must not be called

    async def run():
        store = await _store_with_binding()
        return await _resolve_aliyun_sandbox_env(store, _agent_config(enabled=False), "u1")

    assert asyncio.run(run()) == {}


def test_resolver_skips_when_no_binding(monkeypatch):
    _env(monkeypatch)

    async def run():
        store = InMemoryStore()
        await store.ensure_user("u1")
        return await _resolve_aliyun_sandbox_env(store, _agent_config(), "u1")

    assert asyncio.run(run()) == {}


def test_resolver_is_failure_isolated(monkeypatch):
    """An AssumeRole failure must yield {} (never break sandbox creation)."""
    _env(monkeypatch)

    def boom(*a, **k):
        raise aliyun_sts.AliyunCliError("AssumeRole failed: expired")

    monkeypatch.setattr(aliyun_sts, "assume_role", boom)

    async def run():
        store = await _store_with_binding()
        return await _resolve_aliyun_sandbox_env(store, _agent_config(), "u1")

    assert asyncio.run(run()) == {}


def test_resolver_skips_without_base_creds(monkeypatch):
    _env(monkeypatch, base=False)

    async def run():
        store = await _store_with_binding()
        return await _resolve_aliyun_sandbox_env(store, _agent_config(), "u1")

    assert asyncio.run(run()) == {}


# --------------------------------------------------------------------------- #
# _build_env_contract merge
# --------------------------------------------------------------------------- #
def _fake_provider():
    return types.SimpleNamespace(inject_env_contract=True, extra_envs={})


def test_env_contract_merges_aliyun_env():
    scope = ToolScope(user_id="u1", metadata={"aliyun_sandbox_env": {
        "ALIBABACLOUD_ACCESS_KEY_ID": "STS.x", "ALIBABACLOUD_SECURITY_TOKEN": "t"}})
    envs = _build_env_contract(_fake_provider(), scope, "sess:u1")
    assert envs["ALIBABACLOUD_ACCESS_KEY_ID"] == "STS.x"
    assert envs["ALIBABACLOUD_SECURITY_TOKEN"] == "t"
    assert envs["AGENT_USER_ID"] == "u1"  # base contract still present


def test_env_contract_unchanged_without_aliyun_env():
    scope = ToolScope(user_id="u1", metadata={})
    envs = _build_env_contract(_fake_provider(), scope, "sess:u1")
    assert not any(k.startswith("ALIBABACLOUD_") for k in envs)


def test_mask_body_redacts_injected_sts_secrets():
    """The create-payload logger must never emit the STS secret or token."""
    payload = {"envs": {
        "AGENT_USER_ID": "u1",
        "ALIBABACLOUD_ACCESS_KEY_ID": "STS.Nv123456789",
        "ALIBABACLOUD_ACCESS_KEY_SECRET": "supersecretvalue",
        "ALIBABACLOUD_SECURITY_TOKEN": "CAIStoken....",
    }}
    masked = _mask_body(payload)["envs"]
    assert masked["AGENT_USER_ID"] == "u1"  # non-secret untouched
    assert masked["ALIBABACLOUD_ACCESS_KEY_SECRET"] != "supersecretvalue"
    assert "supersecretvalue" not in str(masked)
    assert "CAIStoken" not in str(masked).replace(".", "")
    # legacy camelCase keys still masked (no regression)
    assert _mask_body({"accessKeySecret": "x", "securityToken": "y", "apiKey": "z"}) \
        != {"accessKeySecret": "x", "securityToken": "y", "apiKey": "z"}
