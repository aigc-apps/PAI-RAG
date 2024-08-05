from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pai_rag.core.models.errors import UserInputError, ServiceError


async def user_input_exception_handler(request: Request, exception: UserInputError):
    return JSONResponse(
        status_code=400,
        content={"message": f"Failed to process request input: {exception.msg}"},
    )


async def service_exception_handler(request: Request, exception: ServiceError):
    return JSONResponse(status_code=500, content={"message": f"Oops, {exception.msg}"})


def config_app_errors(app: FastAPI):
    app.add_exception_handler(UserInputError, user_input_exception_handler)
    app.add_exception_handler(ServiceError, service_exception_handler)
