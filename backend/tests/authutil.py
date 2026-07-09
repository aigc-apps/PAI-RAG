"""Test helper: satisfy the auth dependencies without minting real JWTs.

Every user-facing route now depends on ``require_user`` (and admin routes on
``require_admin``, which itself depends on ``require_user``). Overriding
``require_user`` with a fixed admin ``User`` therefore green-lights both tiers,
so route tests can keep asserting behavior without a login round-trip. Pass
``role="user"`` to exercise the non-admin path.
"""
from __future__ import annotations

from app.auth import require_user
from app.store.base import User

TEST_USER_ID = "u_test"


def apply_auth(app, *, user_id: str = TEST_USER_ID, email: str = "test@example.com",
               role: str = "admin", status: str = "active"):
    app.dependency_overrides[require_user] = lambda: User(
        id=user_id, email=email, role=role, status=status, display_name=None)
    return app
