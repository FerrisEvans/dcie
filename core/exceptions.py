from attrs import define, field
from json import JSONDecodeError

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette import status
from starlette.exceptions import HTTPException

from core.logger import log


class BizException(Exception):
    def __init__(self, message: str = "", code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.code = code

@define(slots=True)
class ReqInfo:
    path: str = field()
    method: str = field()
    body: dict = field(default=None)


async def extract_request_info(request: Request) -> ReqInfo:
    """Extract common request information."""
    request_path = request.url.path
    request_method = request.method
    try:
        request_body = await request.json()
    except JSONDecodeError:
        log.warning(f"Request body cannot decode to JSON: {request_path}")
        request_body = None

    return ReqInfo(path=request_path, method=request_method, body=request_body)


async def log_error(message: str, req: ReqInfo, exc: Exception, **kwargs):
    log.error(f"Request: [{req.method}]//[{req.path}] failed with message: [{message}] "
              f"\nrequest body: [{req.body}]"
              f"\nexception: [{repr(exc)}]"
              f"\nother args: [{kwargs}]")
    log.exception(exc)


def register_exception(app: FastAPI):
    @app.exception_handler(BizException)
    async def biz_exc(request: Request, exc: BizException):
        request_info = await extract_request_info(request)
        await log_error("Business exception occurred", request_info, exc)

        return JSONResponse(status_code=exc.code, content={"message": exc.message})

    @app.exception_handler(HTTPException)
    async def unicorn_exc(request: Request, exc: HTTPException):
        request_info = await extract_request_info(request)
        await log_error("HTTP exception occurred", request_info, exc, detail=exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"message": "HTTP exception occurred"})

    @app.exception_handler(RequestValidationError)
    async def request_validation_exc(request: Request, exc: RequestValidationError):
        request_info = await extract_request_info(request)
        await log_error("Request params validation failed", request_info, exc, detail=exc.errors()[0].get('message'))
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "Request params validation failed"})

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exc(request: Request, exc: SQLAlchemyError):
        request_info = await extract_request_info(request)
        await log_error("Database error occurred", request_info, exc)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "A database error occurred. Please try again later."})

    @app.exception_handler(ResponseValidationError)
    async def response_validation_exc(request: Request, exc: ResponseValidationError):
        request_info = await extract_request_info(request)
        errors = exc.errors()

        # Check if this is a None/null response case
        is_none_response = False
        for error in errors:
            if error.get("input") is None and "valid dictionary" in error.get("msg", ""):
                is_none_response = True
                break

        await log_error("Response validation error occurred", request_info, exc, validation_errors=errors, is_none_response=is_none_response)

        if is_none_response:
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND,content={"no_response": "The requested resource was not found"},)
        else:
            return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"response_format_error": errors})

    @app.exception_handler(ValueError)
    async def value_exc(request: Request, exc: ValueError):
        request_info = await extract_request_info(request)
        await log_error("Value error", request_info, exc)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": status.HTTP_400_BAD_REQUEST.__str__()})

    @app.exception_handler(Exception)
    async def global_exc(request: Request, exc: Exception):
        request_info = await extract_request_info(request)
        await log_error("Unexpected Exception", request_info, exc)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": status.HTTP_500_INTERNAL_SERVER_ERROR.__str__()})
