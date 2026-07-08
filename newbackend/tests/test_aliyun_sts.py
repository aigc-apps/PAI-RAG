import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent.integrations import aliyun_sts as sts


# --------------------------------------------------------------------------- #
# ExternalId
# --------------------------------------------------------------------------- #
def test_external_id_is_deterministic_per_user_and_secret():
    a = sts.derive_external_id("u_123", secret="s3cret")
    b = sts.derive_external_id("u_123", secret="s3cret")
    assert a == b
    assert a.startswith("pai-") and len(a) == len("pai-") + 32


def test_external_id_varies_by_user_and_secret():
    assert sts.derive_external_id("u_1", secret="k") != sts.derive_external_id("u_2", secret="k")
    assert sts.derive_external_id("u_1", secret="k1") != sts.derive_external_id("u_1", secret="k2")


def test_external_id_fails_closed_without_secret():
    with pytest.raises(ValueError):
        sts.derive_external_id("u_1", secret="")


# --------------------------------------------------------------------------- #
# Per-user role name (so multiple users on one account don't collide)
# --------------------------------------------------------------------------- #
def test_role_name_is_deterministic_and_per_user():
    a = sts.derive_role_name("u_123", secret="s3cret")
    assert a == sts.derive_role_name("u_123", secret="s3cret")
    # hash-only when no email: prefix + 12 hex
    assert a.startswith("pai-agent-") and len(a) == len("pai-agent-") + 12
    assert sts.valid_role_name(a)
    # distinct per user and per secret
    assert sts.derive_role_name("u_1", secret="k") != sts.derive_role_name("u_2", secret="k")
    assert sts.derive_role_name("u_1", secret="k1") != sts.derive_role_name("u_1", secret="k2")


def test_role_name_embeds_readable_email_slug():
    # Email local part becomes a legible, RAM-safe label; hash still disambiguates.
    n = sts.derive_role_name("u_123", secret="s3cret", email="Alice.Wong+test@corp.com")
    assert n.startswith("pai-agent-alice-wong-test-")
    assert sts.valid_role_name(n) and len(n) <= 64
    # deterministic, and the email is only a label — the hash suffix still tracks user_id
    assert n == sts.derive_role_name("u_123", secret="s3cret", email="alice.wong+test@corp.com")
    assert n.endswith(sts.derive_role_name("u_123", secret="s3cret")[len("pai-agent-"):])
    # a wildly long / messy local part stays within RAM's 64-char limit
    long = sts.derive_role_name("u_9", secret="k", email=("x" * 80) + "@e.com")
    assert sts.valid_role_name(long) and len(long) <= 64


def test_role_name_independent_of_external_id():
    # Same user id, but the two derivations must not coincide (namespaced HMAC).
    uid, secret = "u_9", "k"
    assert sts.derive_role_name(uid, secret=secret)[len("pai-agent-"):] != \
        sts.derive_external_id(uid, secret=secret)[len("pai-"):][:12]


def test_role_name_honors_prefix_and_fails_closed():
    assert sts.derive_role_name("u_1", secret="k", prefix="acme-").startswith("acme-")
    with pytest.raises(ValueError):
        sts.derive_role_name("u_1", secret="")
    with pytest.raises(ValueError):
        sts.derive_role_name("", secret="k")


@pytest.mark.parametrize("name,ok", [
    ("pai-agent-abcdef0123456789", True),
    ("pai-agent-crossaccount-role", True),
    ("", False),
    ("bad name with spaces", False),
    ("inject;rm", False),
    ("x" * 65, False),
])
def test_valid_role_name(name, ok):
    assert sts.valid_role_name(name) is ok


# --------------------------------------------------------------------------- #
# RoleArn validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arn", [
    "acs:ram::1730760139076263:role/feiyue-test-role",
    "acs:ram::1095312831785714:role/pai_authz-Role.1",
])
def test_valid_role_arn_accepts(arn):
    assert sts.valid_role_arn(arn)


@pytest.mark.parametrize("arn", [
    "",
    "acs:ram::123:role/short-account",              # account not 16 digits
    "acs:ram::1730760139076263:role/bad;rm -rf",    # injection chars
    "acs:ram::1730760139076263:user/not-a-role",
    "arn:aws:iam::1730760139076263:role/x",         # wrong partition
])
def test_valid_role_arn_rejects(arn):
    assert not sts.valid_role_arn(arn)


# --------------------------------------------------------------------------- #
# AssumeRole command construction
# --------------------------------------------------------------------------- #
def test_assume_role_uses_isolated_config_off_argv_and_env(monkeypatch):
    captured = {}

    def fake_run(cmd, env, timeout):
        captured["cmd"] = cmd
        captured["env"] = env
        # the throwaway config still exists during the call → capture its contents
        i = cmd.index("--config-path")
        captured["config"] = json.load(open(cmd[i + 1]))
        out = (
            '{"Credentials": {"AccessKeyId": "STS.x", "AccessKeySecret": "sec",'
            ' "SecurityToken": "tok", "Expiration": "2026-07-07T05:00:00Z"},'
            ' "AssumedRoleUser": {"Arn": "acs:ram::17:role/r/sess"}}'
        )
        return 0, out, ""

    # An ambient session token must never leak into the base AssumeRole call.
    monkeypatch.setenv("ALIBABACLOUD_SECURITY_TOKEN", "leftover-should-be-removed")
    monkeypatch.setenv("ALIBABACLOUD_ACCESS_KEY_ID", "ambient-should-be-removed")
    monkeypatch.setattr(sts, "_run", fake_run)

    creds = sts.assume_role(
        "acs:ram::1730760139076263:role/feiyue-test-role", "pai-abc",
        region="cn-hangzhou", base_ak="AK", base_sk="SK", account_id="1095312831785714",
    )

    cmd = captured["cmd"]
    assert cmd[:3] == ["aliyun", "sts", "AssumeRole"]
    assert "--config-path" in cmd
    assert "--RoleArn" in cmd and "acs:ram::1730760139076263:role/feiyue-test-role" in cmd
    assert "--RoleSessionName" in cmd and "--ExternalId" in cmd and "pai-abc" in cmd
    assert "--DurationSeconds" in cmd and "3600" in cmd
    # secret is neither on argv (procfs-visible) nor in the child env
    assert "SK" not in cmd
    assert all("SK" not in str(v) for v in captured["env"].values())
    assert "ALIBABACLOUD_SECURITY_TOKEN" not in captured["env"]
    assert "ALIBABACLOUD_ACCESS_KEY_ID" not in captured["env"]
    # the isolated config carries the developer AK/SK under an AK profile
    prof = captured["config"]["profiles"][0]
    assert prof["mode"] == "AK" and prof["access_key_id"] == "AK" and prof["access_key_secret"] == "SK"
    assert captured["config"]["current"] == prof["name"]
    assert creds.access_key_id == "STS.x" and creds.security_token == "tok"
    assert creds.assumed_role_arn == "acs:ram::17:role/r/sess"


def test_assume_role_rejects_bad_arn():
    with pytest.raises(ValueError):
        sts.assume_role("not-an-arn", "pai-abc", region="cn-hangzhou",
                        base_ak="AK", base_sk="SK")


def test_assume_role_raises_with_parsed_error(monkeypatch):
    err = "\x1b[1;31mERROR: SDK.ServerError\nErrorCode: NoPermission\nMessage: not allowed\n\x1b[0m"
    monkeypatch.setattr(sts, "_run", lambda cmd, env, timeout: (1, "", err))
    with pytest.raises(sts.AliyunCliError) as ei:
        sts.assume_role("acs:ram::1730760139076263:role/r", "pai-abc",
                        region="cn-hangzhou", base_ak="AK", base_sk="SK")
    assert "NoPermission" in str(ei.value)


def test_configured_assume_duration_defaults_to_12h_and_clamps():
    # Default (nothing configured) is the 12h ceiling now, not the old 1h.
    assert sts.configured_assume_duration_seconds(None) == 43200
    assert sts.configured_assume_duration_seconds({}) == 43200
    # Explicit values clamp into [3600, 43200].
    assert sts.configured_assume_duration_seconds({"assume_duration_seconds": 7200}) == 7200
    assert sts.configured_assume_duration_seconds({"assume_duration_seconds": 99999}) == 43200
    assert sts.configured_assume_duration_seconds({"assume_duration_seconds": 60}) == 900


def test_assume_role_falls_back_to_3600_when_role_rejects_long_duration(monkeypatch):
    """A role whose MaxSessionDuration predates the 12h bump rejects a 43200s
    request; assume_role retries once at the 3600s floor rather than failing."""
    durations = []

    def fake_run(cmd, env, timeout):
        d = cmd[cmd.index("--DurationSeconds") + 1]
        durations.append(d)
        if d != "3600":
            return (1, "", "ErrorCode: InvalidParameter\nMessage: The parameter "
                            "DurationSeconds is out of range for this role")
        out = ('{"Credentials": {"AccessKeyId": "STS.x", "AccessKeySecret": "s",'
               ' "SecurityToken": "tok", "Expiration": "2026-07-08T05:00:00Z"}}')
        return (0, out, "")

    monkeypatch.setattr(sts, "_run", fake_run)
    creds = sts.assume_role("acs:ram::1730760139076263:role/r", "pai-abc", region="cn-hangzhou",
                            base_ak="AK", base_sk="SK", duration_seconds=43200)
    assert durations == ["43200", "3600"]  # tried long first, then fell back
    assert creds.security_token == "tok"


def test_assume_role_does_not_retry_on_unrelated_error(monkeypatch):
    calls = []

    def fake_run(cmd, env, timeout):
        calls.append(1)
        return (1, "", "ErrorCode: NoPermission\nMessage: not allowed")

    monkeypatch.setattr(sts, "_run", fake_run)
    with pytest.raises(sts.AliyunCliError):
        sts.assume_role("acs:ram::1730760139076263:role/r", "pai-abc", region="cn-hangzhou",
                        base_ak="AK", base_sk="SK", duration_seconds=43200)
    assert len(calls) == 1  # no fallback retry for a non-duration failure


# --------------------------------------------------------------------------- #
# verify_pai_access verdict parsing
# --------------------------------------------------------------------------- #
_CREDS = sts.Credentials(access_key_id="STS.x", access_key_secret="s", security_token="t")


def test_verify_success(monkeypatch):
    calls = []

    def fake_run(cmd, env, timeout):
        calls.append(cmd)
        if cmd[1:3] == ["sts", "GetCallerIdentity"]:
            return 0, '{"AccountId": "1730760139076263", "Arn": "acs:ram::17:assumed-role/r/s"}', ""
        return 0, '{"TotalCount": 2, "Services": [{"ServiceName": "a"}, {"ServiceName": "b"}]}', ""

    monkeypatch.setattr(sts, "_run", fake_run)
    v = sts.verify_pai_access(_CREDS, regions=["cn-hangzhou"])
    assert v.ok and v.stage == "pai"
    assert v.account_id == "1730760139076263"
    assert v.pai_total == 2
    assert len(calls) == 2


def test_verify_identity_expired(monkeypatch):
    err = "ErrorCode: InvalidSecurityToken.Expired\nMessage: Specified SecurityToken is expired."
    monkeypatch.setattr(sts, "_run", lambda cmd, env, timeout: (1, "", err))
    v = sts.verify_pai_access(_CREDS, regions=["cn-hangzhou"])
    assert not v.ok and v.stage == "identity"
    assert v.error_code == "InvalidSecurityToken.Expired"


def test_verify_pai_access_denied(monkeypatch):
    def fake_run(cmd, env, timeout):
        if cmd[1:3] == ["sts", "GetCallerIdentity"]:
            return 0, '{"AccountId": "17", "Arn": "acs:ram::17:assumed-role/r/s"}', ""
        return 1, "", "ErrorCode: NoPermission\nMessage: no eas access"

    monkeypatch.setattr(sts, "_run", fake_run)
    v = sts.verify_pai_access(_CREDS, regions=["cn-hangzhou"])
    assert not v.ok and v.stage == "pai"
    assert v.account_id == "17" and v.error_code == "NoPermission"


def test_verify_pai_malformed_json_is_tolerated(monkeypatch):
    def fake_run(cmd, env, timeout):
        if cmd[1:3] == ["sts", "GetCallerIdentity"]:
            return 0, '{"AccountId": "17"}', ""
        return 0, "not json", ""

    monkeypatch.setattr(sts, "_run", fake_run)
    v = sts.verify_pai_access(_CREDS, regions=["cn-hangzhou"])
    assert v.ok and v.pai_total is None


def test_verify_probes_multiple_regions_and_aggregates(monkeypatch):
    def fake_run(cmd, env, timeout):
        if cmd[1:3] == ["sts", "GetCallerIdentity"]:
            return 0, '{"AccountId": "17", "Arn": "acs:ram::17:assumed-role/r/s"}', ""
        # endpoint carries the region: pai-eas.<region>.aliyuncs.com
        endpoint = cmd[cmd.index("--endpoint") + 1]
        total = 2 if "cn-shanghai" in endpoint else 0
        return 0, f'{{"TotalCount": {total}}}', ""

    monkeypatch.setattr(sts, "_run", fake_run)
    v = sts.verify_pai_access(_CREDS, regions=["cn-hangzhou", "cn-shanghai"])
    assert v.ok and v.pai_total == 2
    assert [r["region"] for r in v.regions] == ["cn-hangzhou", "cn-shanghai"]
    shanghai = next(r for r in v.regions if r["region"] == "cn-shanghai")
    assert shanghai["ok"] and shanghai["pai_total"] == 2


def test_verify_ok_when_one_region_fails_but_another_succeeds(monkeypatch):
    def fake_run(cmd, env, timeout):
        if cmd[1:3] == ["sts", "GetCallerIdentity"]:
            return 0, '{"AccountId": "17"}', ""
        endpoint = cmd[cmd.index("--endpoint") + 1]
        if "cn-hangzhou" in endpoint:
            return 1, "", "ErrorCode: NoPermission\nMessage: nope"
        return 0, '{"TotalCount": 1}', ""

    monkeypatch.setattr(sts, "_run", fake_run)
    v = sts.verify_pai_access(_CREDS, regions=["cn-hangzhou", "cn-shanghai"])
    assert v.ok and v.pai_total == 1
    hangzhou = next(r for r in v.regions if r["region"] == "cn-hangzhou")
    assert not hangzhou["ok"] and hangzhou["error_code"] == "NoPermission"


def test_verify_fails_only_when_all_regions_error(monkeypatch):
    def fake_run(cmd, env, timeout):
        if cmd[1:3] == ["sts", "GetCallerIdentity"]:
            return 0, '{"AccountId": "17"}', ""
        return 1, "", "ErrorCode: NoPermission\nMessage: nope"

    monkeypatch.setattr(sts, "_run", fake_run)
    v = sts.verify_pai_access(_CREDS, regions=["cn-hangzhou", "cn-shanghai"])
    assert not v.ok and v.stage == "pai" and v.error_code == "NoPermission"


def test_verify_uses_isolated_sts_config_off_argv(monkeypatch):
    seen = {}

    def fake_run(cmd, env, timeout):
        if cmd[1:3] == ["sts", "GetCallerIdentity"]:
            i = cmd.index("--config-path")
            seen["config"] = json.load(open(cmd[i + 1]))
            seen["id_cmd"] = cmd
            return 0, '{"AccountId": "17"}', ""
        return 0, '{"TotalCount": 0}', ""

    monkeypatch.setattr(sts, "_run", fake_run)
    creds = sts.Credentials(access_key_id="STS.x", access_key_secret="shh", security_token="TOK")
    v = sts.verify_pai_access(creds, regions=["cn-hangzhou"])
    assert v.ok
    prof = seen["config"]["profiles"][0]
    assert prof["mode"] == "StsToken"
    assert prof["access_key_id"] == "STS.x" and prof["sts_token"] == "TOK"
    # session secret/token stay off argv (procfs-visible)
    assert "shh" not in seen["id_cmd"] and "TOK" not in seen["id_cmd"]


def test_configured_regions_default_and_explicit():
    # explicit list wins verbatim (de-duped, order preserved)
    assert sts.configured_regions({"regions": ["cn-shanghai", "cn-shanghai", "us-west-1"]},
                                  "cn-hangzhou") == ["cn-shanghai", "us-west-1"]
    # otherwise the configured single region is pinned first
    default = sts.configured_regions({"region": "cn-shenzhen"}, "cn-hangzhou")
    assert default[0] == "cn-shenzhen" and "cn-hangzhou" in default


def test_render_ros_template_substitutes_account_id():
    out = sts.render_ros_template("1122334455667788")
    assert "acs:ram::1122334455667788:root" in out
    assert "${DEVELOPER_ACCOUNT_ID}" not in out
    # No per-user values → placeholders resolve to blank / the generic default.
    assert "${EXTERNAL_ID}" not in out and "${ROLE_NAME}" not in out
    assert "pai-agent-crossaccount-role" in out


def test_render_ros_template_bakes_per_user_defaults():
    out = sts.render_ros_template(
        "1122334455667788",
        external_id="pai-159a4f64385292c3cafa867954672314",
        role_name="pai-agent-feiyue-c7ab388b5613",
    )
    # Both land as the parameters' Default so the ROS create form is pre-filled.
    assert "Default: 'pai-159a4f64385292c3cafa867954672314'" in out
    assert "Default: 'pai-agent-feiyue-c7ab388b5613'" in out
    assert "${EXTERNAL_ID}" not in out and "${ROLE_NAME}" not in out


def test_render_ros_template_rejects_unsafe_per_user_values():
    # App-generated identifiers only; anything that could break the YAML scalar
    # is rejected rather than inlined.
    with pytest.raises(ValueError):
        sts.render_ros_template("112233", external_id="x' bad: true #")
    with pytest.raises(ValueError):
        sts.render_ros_template("112233", role_name="has space")


def test_render_ros_template_fails_without_account_id():
    with pytest.raises(ValueError):
        sts.render_ros_template("")


def test_parse_cli_error_strips_ansi():
    code, msg = sts.parse_cli_error("\x1b[1;31mErrorCode: X.Y\nMessage: boom\x1b[0m")
    assert code == "X.Y" and msg == "boom"


@pytest.mark.parametrize("code,message,expected", [
    ("InvalidSecurityToken.Expired", "The security token is expired", "credential"),
    ("InvalidAccessKeyId.NotFound", "", "credential"),
    ("SignatureDoesNotMatch", "", "credential"),
    ("MissingSecurityToken", "", "credential"),
    ("", "no credential found, please run `aliyun configure`", "credential"),
    # Unbound state: the CLI has no profile file at all.
    ("", "load configure failed: stat /home/user/.aliyun/config.json: no such file or directory", "credential"),
    ("NoPermission", "You are not authorized to do this", "permission"),
    ("Forbidden.RAM", "no permission", "permission"),
    ("InvalidRegionId", "region not found", "other"),
    ("EntityNotExist.Role", "", "other"),
    ("", "", "other"),
])
def test_classify_cli_error(code, message, expected):
    assert sts.classify_cli_error(code, message) == expected


def test_classify_permission_beats_credential_on_overlap():
    # "unauthorized" appears in both worlds; a genuine permission denial must not
    # be misread as a credential problem (which would trigger a re-auth loop).
    assert sts.classify_cli_error(
        "NoPermission", "You are not authorized to operate aliyun configure") == "permission"


# --------------------------------------------------------------------------- #
# base-cred resolution (single source shared by route / builder / status)
# --------------------------------------------------------------------------- #
import types


def _cfg(settings):
    return types.SimpleNamespace(
        providers=[types.SimpleNamespace(id="aliyun_pai.default", settings=settings)])


def test_base_cred_env_names_default_and_override():
    assert sts.base_cred_env_names(None) == ("AGENTRUN_ACCESS_KEY_ID", "AGENTRUN_ACCESS_KEY_SECRET")
    assert sts.base_cred_env_names({"base_access_key_id_env": "MY_AK",
                                    "base_access_key_secret_env": "MY_SK"}) == ("MY_AK", "MY_SK")


def test_read_base_creds_honors_custom_env_names():
    env = {"MY_AK": "ak", "MY_SK": "sk", "AGENTRUN_ACCESS_KEY_ID": "wrong"}
    settings = {"base_access_key_id_env": "MY_AK", "base_access_key_secret_env": "MY_SK"}
    assert sts.read_base_creds(settings, environ=env) == ("ak", "sk")
    # default names when unset
    assert sts.read_base_creds(None, environ={"AGENTRUN_ACCESS_KEY_ID": "a",
                                              "AGENTRUN_ACCESS_KEY_SECRET": "b"}) == ("a", "b")


def test_provider_settings_extracts_block():
    assert sts.provider_settings(_cfg({"base_access_key_id_env": "X"})) == {"base_access_key_id_env": "X"}
    assert sts.provider_settings(types.SimpleNamespace(providers=[])) == {}
    assert sts.provider_settings(None) == {}
