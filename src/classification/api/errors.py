"""OpenAI-style error envelope: {"error": {"message", "type", "param", "code"}}."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    404: "not_found_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    503: "service_unavailable",
}


class APIError(Exception):
    def __init__(self, status: int, message: str, param: str | None = None, code: str | None = None):
        self.status, self.message, self.param, self.code = status, message, param, code


def _body(status: int, message: str, param: str | None = None, code: str | None = None) -> dict:
    return {"error": {"message": message, "type": _TYPES.get(status, "api_error"), "param": param, "code": code}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        return JSONResponse(_body(exc.status, exc.message, exc.param, exc.code), status_code=exc.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(_body(exc.status_code, str(exc.detail)), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        errs = exc.errors()
        first = errs[0] if errs else {}
        param = ".".join(str(p) for p in first.get("loc", ())[1:]) or None
        return JSONResponse(_body(422, first.get("msg", "Invalid request"), param, "invalid_value"), status_code=422)
