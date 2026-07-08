import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException

from app.auth import (
    ALGORITHM,
    create_access_token,
    decode_token,
    hash_password,
    hash_token,
    invite_expired,
    new_invite_token,
    verify_password,
)
from app.config import Settings
from app.store.base import User, UserAuth


def _settings(**over):
    base = dict(jwt_secret="unit-test-secret", jwt_ttl_minutes=720, invite_ttl_hours=72)
    base.update(over)
    return Settings(**base)


def _user(**over):
    d = dict(id="u1", email="a@b.com", role="user", status="active", display_name=None)
    d.update(over)
    return User(**d)


# --------------------------------------------------------------------------- #
# passwords
# --------------------------------------------------------------------------- #
def test_hash_password_roundtrip_and_wrong():
    h = hash_password("s3kr3t-passphrase")
    assert h and h != "s3kr3t-passphrase"          # never stored in the clear
    assert verify_password("s3kr3t-passphrase", h) is True
    assert verify_password("nope", h) is False


def test_verify_password_false_on_absent_or_garbage_hash():
    assert verify_password("x", None) is False
    assert verify_password("x", "") is False
    assert verify_password("x", "not-a-real-argon2-hash") is False


# --------------------------------------------------------------------------- #
# JWT access tokens
# --------------------------------------------------------------------------- #
def test_token_roundtrip_carries_claims():
    s = _settings()
    tok = create_access_token(_user(role="admin"), settings=s)
    claims = decode_token(tok, settings=s)
    assert claims["sub"] == "u1"
    assert claims["email"] == "a@b.com"
    assert claims["role"] == "admin"
    assert claims["exp"] > claims["iat"]


def test_expired_token_raises_401():
    s = _settings(jwt_ttl_minutes=-1)              # exp already in the past
    tok = create_access_token(_user(), settings=s)
    with pytest.raises(HTTPException) as ei:
        decode_token(tok, settings=_settings())
    assert ei.value.status_code == 401
    assert "expired" in ei.value.detail


def test_tampered_token_raises_401():
    s = _settings()
    tok = create_access_token(_user(), settings=s)
    # flip a character in the signature segment
    head, payload, sig = tok.split(".")
    bad = f"{head}.{payload}.{sig[:-2]}xx"
    with pytest.raises(HTTPException) as ei:
        decode_token(bad, settings=s)
    assert ei.value.status_code == 401


def test_wrong_secret_raises_401():
    tok = create_access_token(_user(), settings=_settings(jwt_secret="secret-A"))
    with pytest.raises(HTTPException) as ei:
        decode_token(tok, settings=_settings(jwt_secret="secret-B"))
    assert ei.value.status_code == 401


def test_wrong_alg_is_rejected():
    s = _settings()
    # A token signed with a different HMAC alg must not verify under HS256-only.
    forged = jwt.encode({"sub": "u1"}, s.jwt_secret, algorithm="HS512")
    with pytest.raises(HTTPException) as ei:
        decode_token(forged, settings=s)
    assert ei.value.status_code == 401


def test_alg_none_is_rejected():
    # Classic "alg:none" downgrade — an unsigned token must be rejected.
    forged = jwt.encode({"sub": "u1"}, key="", algorithm="none")
    with pytest.raises(HTTPException) as ei:
        decode_token(forged, settings=_settings())
    assert ei.value.status_code == 401


def test_unconfigured_secret_raises_503_on_decode_and_runtimeerror_on_issue():
    s = _settings(jwt_secret="")
    with pytest.raises(HTTPException) as ei:
        decode_token("whatever", settings=s)
    assert ei.value.status_code == 503
    with pytest.raises(RuntimeError):
        create_access_token(_user(), settings=s)


# --------------------------------------------------------------------------- #
# invite tokens
# --------------------------------------------------------------------------- #
def test_new_invite_token_hash_matches_and_is_opaque():
    raw, h = new_invite_token()
    assert raw and h and raw != h
    assert hash_token(raw) == h                    # server stores only the hash
    # a different raw yields a different hash
    raw2, h2 = new_invite_token()
    assert h2 != h


def _auth(exp):
    return UserAuth(id="u1", email="a@b.com", role="user", status="invited",
                    password_hash=None, invite_token_hash="h", invite_expires_at=exp)


def test_invite_expired_semantics():
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    assert invite_expired(_auth(future)) is False
    assert invite_expired(_auth(past)) is True
    assert invite_expired(_auth(None)) is True     # missing expiry is treated as expired


def test_invite_expired_handles_naive_datetime():
    # A naive (tz-less) timestamp coming back from the DB is assumed UTC.
    naive_future = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(tzinfo=None)
    assert invite_expired(_auth(naive_future)) is False
