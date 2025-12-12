# exceptions for code sandbox


class CodeSandboxException(Exception):
    """Base exception for code sandbox errors"""
    pass


class CodeSandboxEmptyCodeException(CodeSandboxException):
    """Exception raised when code is empty or invalid"""
    pass


class CodeSandboxNotInitializedException(CodeSandboxException):
    """Exception raised when session or context is not initialized"""
    pass


class CodeSandboxTimeoutException(CodeSandboxException):
    """Exception raised when execution times out"""
    pass


class CodeSandboxHTTPException(CodeSandboxException):
    """Exception raised when HTTP request fails"""
    pass


class CodeSandboxAPIException(CodeSandboxException):
    """Exception raised when API returns non-success code"""
    pass


class CodeSandboxExecutionException(CodeSandboxException):
    """Exception raised during code execution"""
    pass


class CodeSandboxNotConfiguredException(CodeSandboxException):
    """Exception raised when code sandbox is not configured"""
    pass
