"""Short, collision-safe id generator (`new_id`) + the collision-retry wrapper.

Shape: ``prefix_`` + 11 Base58 chars (58**11 ≈ 2**64.4 entropy). Base58 omits
the visually/LLM-ambiguous ``0 O I l`` so a model can echo an id back as a tool
argument without inviting a mis-copied character. The DB unique index is the
correctness backstop; ``with_id_retry`` re-runs the write on the ~never collision.
"""

import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy.exc import IntegrityError

from app.store.base import _ID_ALPHABET, new_id, with_id_retry, _uuid


DOC_ID = re.compile(r"^doc_[1-9A-HJ-NP-Za-km-z]{11}$")


def test_new_id_shape_and_length():
    ident = new_id("doc")
    assert DOC_ID.match(ident), ident
    assert len(ident) == 15  # "doc_" (4) + 11


def test_alphabet_excludes_ambiguous_chars():
    for ch in "0OIl":
        assert ch not in _ID_ALPHABET
    assert len(_ID_ALPHABET) == 58


def test_uuid_is_new_id_alias():
    assert _uuid is new_id
    assert DOC_ID.match(_uuid("doc"))


def test_ids_are_distinct():
    ids = {new_id("chk") for _ in range(1000)}
    assert len(ids) == 1000


def test_with_id_retry_recovers_after_transient_collisions():
    async def scenario():
        calls = {"n": 0}

        async def work():
            calls["n"] += 1
            if calls["n"] < 3:
                raise IntegrityError("stmt", {}, Exception("UNIQUE constraint failed"))
            return "ok"

        result = await with_id_retry(work)  # default attempts=3
        return result, calls["n"]

    result, n = asyncio.run(scenario())
    assert result == "ok"
    assert n == 3


def test_with_id_retry_reraises_when_exhausted():
    async def scenario():
        async def work():
            raise IntegrityError("stmt", {}, Exception("UNIQUE constraint failed"))

        with pytest.raises(IntegrityError):
            await with_id_retry(work, attempts=3)

    asyncio.run(scenario())


def test_with_id_retry_does_not_swallow_other_errors():
    async def scenario():
        async def work():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await with_id_retry(work)

    asyncio.run(scenario())
