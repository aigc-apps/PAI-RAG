"""Tests for the unified API exception handling layer.

Covers:
    - `handle_api_exceptions` decorator: pass-through, ValueError /
      ValidationError -> 400, generic Exception -> 500, custom codes,
      action wording in messages, HTTPException pass-through.
    - `unhandled_exception_handler`: shape of the JSON body.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import json
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

from api.api_exception import (
    ApiException,
    handle_api_exceptions,
    unhandled_exception_handler,
)


class TestHandleApiExceptionsPassThrough:
    async def test_returns_result_unchanged(self):
        @handle_api_exceptions(action="do thing")
        async def ok():
            return {"hello": "world"}

        assert await ok() == {"hello": "world"}

    async def test_api_exception_is_reraised_as_is(self):
        @handle_api_exceptions()
        async def fail():
            raise ApiException(code=404, message="missing", data={"id": "x"})

        with pytest.raises(ApiException) as exc:
            await fail()
        assert exc.value.status_code == 404
        assert exc.value.detail == "missing"
        assert exc.value.data == {"id": "x"}

    async def test_http_exception_is_reraised_as_is(self):
        @handle_api_exceptions()
        async def fail():
            raise HTTPException(status_code=418, detail="teapot")

        with pytest.raises(HTTPException) as exc:
            await fail()
        assert exc.value.status_code == 418
        assert exc.value.detail == "teapot"
        # And it must NOT have been wrapped in ApiException
        assert not isinstance(exc.value, ApiException)


class TestHandleApiExceptionsCoercion:
    async def test_value_error_becomes_400(self):
        @handle_api_exceptions(action="create kb")
        async def fail():
            raise ValueError("kb name is required")

        with pytest.raises(ApiException) as exc:
            await fail()
        assert exc.value.status_code == 400
        assert "kb name is required" in exc.value.detail

    async def test_pydantic_validation_error_becomes_400(self):
        class _Model(BaseModel):
            count: int = Field(..., gt=0)

        @handle_api_exceptions()
        async def fail():
            _Model(count=-1)  # raises ValidationError

        with pytest.raises(ApiException) as exc:
            await fail()
        assert exc.value.status_code == 400

    async def test_unknown_exception_becomes_500(self):
        @handle_api_exceptions(action="frobnicate widget")
        async def fail():
            raise RuntimeError("boom")

        with pytest.raises(ApiException) as exc:
            await fail()
        assert exc.value.status_code == 500
        assert "frobnicate widget" in exc.value.detail
        assert "boom" in exc.value.detail

    async def test_action_defaults_to_function_name(self):
        @handle_api_exceptions()
        async def list_things():
            raise RuntimeError("nope")

        with pytest.raises(ApiException) as exc:
            await list_things()
        assert "list things" in exc.value.detail


class TestHandleApiExceptionsCustomCodes:
    async def test_custom_value_error_code(self):
        @handle_api_exceptions(value_error_code=422)
        async def fail():
            raise ValueError("bad")

        with pytest.raises(ApiException) as exc:
            await fail()
        assert exc.value.status_code == 422

    async def test_custom_default_code(self):
        @handle_api_exceptions(default_code=503)
        async def fail():
            raise RuntimeError("downstream gone")

        with pytest.raises(ApiException) as exc:
            await fail()
        assert exc.value.status_code == 503


class TestHandleApiExceptionsPreservesMetadata:
    async def test_preserves_function_name_and_doc(self):
        @handle_api_exceptions()
        async def my_endpoint():
            """The docstring."""
            return 1

        assert my_endpoint.__name__ == "my_endpoint"
        assert my_endpoint.__doc__ == "The docstring."

    async def test_chains_original_exception(self):
        @handle_api_exceptions()
        async def fail():
            raise ValueError("root cause")

        with pytest.raises(ApiException) as exc:
            await fail()
        assert isinstance(exc.value.__cause__, ValueError)
        assert str(exc.value.__cause__) == "root cause"


class TestUnhandledExceptionHandler:
    async def test_returns_500_with_structured_body(self):
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/v1/whatever"

        response = await unhandled_exception_handler(
            request, RuntimeError("kaboom")
        )

        assert response.status_code == 500
        body = json.loads(response.body)
        assert body == {
            "code": 500,
            "message": "Internal server error: RuntimeError",
            "data": None,
        }

    async def test_includes_exception_type_name_not_value(self):
        """The body must not leak `str(exc)` to the client; only the class name.

        This guards against accidentally exposing internal details (file paths,
        stack-trace-like reprs, etc.) in production error responses.
        """
        request = MagicMock()
        request.method = "POST"
        request.url.path = "/v1/x"

        class _Secret(Exception):
            pass

        response = await unhandled_exception_handler(
            request, _Secret("DB password=hunter2")
        )
        body = json.loads(response.body)
        assert "hunter2" not in body["message"]
        assert "_Secret" in body["message"]
