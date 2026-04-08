import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from tools.code.code_sandbox_exceptions import (
    CodeSandboxException,
    CodeSandboxEmptyCodeException,
    CodeSandboxNotInitializedException,
    CodeSandboxTimeoutException,
    CodeSandboxHTTPException,
    CodeSandboxAPIException,
    CodeSandboxExecutionException,
    CodeSandboxNotConfiguredException,
)


class TestCodeSandboxExceptions:
    def test_base_exception(self):
        exc = CodeSandboxException("base error")
        assert str(exc) == "base error"
        assert isinstance(exc, Exception)

    def test_empty_code_exception(self):
        exc = CodeSandboxEmptyCodeException("empty code")
        assert isinstance(exc, CodeSandboxException)

    def test_not_initialized_exception(self):
        exc = CodeSandboxNotInitializedException()
        assert isinstance(exc, CodeSandboxException)

    def test_timeout_exception(self):
        exc = CodeSandboxTimeoutException("timed out")
        assert isinstance(exc, CodeSandboxException)

    def test_http_exception(self):
        exc = CodeSandboxHTTPException("http error")
        assert isinstance(exc, CodeSandboxException)

    def test_api_exception(self):
        exc = CodeSandboxAPIException("api error")
        assert isinstance(exc, CodeSandboxException)

    def test_execution_exception(self):
        exc = CodeSandboxExecutionException("exec error")
        assert isinstance(exc, CodeSandboxException)

    def test_not_configured_exception(self):
        exc = CodeSandboxNotConfiguredException("not configured")
        assert isinstance(exc, CodeSandboxException)

    def test_all_catchable_by_base(self):
        exceptions = [
            CodeSandboxEmptyCodeException,
            CodeSandboxNotInitializedException,
            CodeSandboxTimeoutException,
            CodeSandboxHTTPException,
            CodeSandboxAPIException,
            CodeSandboxExecutionException,
            CodeSandboxNotConfiguredException,
        ]
        for exc_cls in exceptions:
            try:
                raise exc_cls("test")
            except CodeSandboxException:
                pass  # should be caught

    def test_raise_and_catch(self):
        import pytest
        with pytest.raises(CodeSandboxTimeoutException):
            raise CodeSandboxTimeoutException("timeout!")
