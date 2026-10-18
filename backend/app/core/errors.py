"""Domain errors and the single place that maps every error to a problem+json response."""

from http import HTTPStatus

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.core.middleware import REQUEST_ID_HEADER
from app.schemas.problem import PROBLEM_MEDIA_TYPE, Problem

log = structlog.get_logger(__name__)


class AppError(Exception):
    """Base for errors services raise on purpose; subclasses pick status and title."""

    status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR
    title: str = "Internal Server Error"

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.title)
        self.detail = detail


class NotFoundError(AppError):
    status = HTTPStatus.NOT_FOUND
    title = "Not Found"


class ConflictError(AppError):
    status = HTTPStatus.CONFLICT
    title = "Conflict"


def problem_response(
    request: Request,
    status: int,
    *,
    title: str | None = None,
    detail: str | None = None,
    **extensions: object,
) -> JSONResponse:
    """Build a problem+json response for `request`."""
    problem = Problem.model_validate(
        {
            "title": title or HTTPStatus(status).phrase,
            "status": status,
            "detail": detail,
            "instance": request.url.path,
            "request_id": getattr(request.state, "request_id", None),
            **extensions,
        }
    )
    return JSONResponse(
        problem.model_dump(mode="json", exclude_none=True),
        status_code=status,
        media_type=PROBLEM_MEDIA_TYPE,
    )


async def _app_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    return problem_response(request, exc.status, title=exc.title, detail=exc.detail)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    detail = exc.detail if exc.detail != HTTPStatus(exc.status_code).phrase else None
    response = problem_response(request, exc.status_code, detail=detail)
    if exc.headers:
        response.headers.update(exc.headers)
    return response


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return problem_response(
        request,
        HTTPStatus.UNPROCESSABLE_CONTENT,
        detail="The request is invalid.",
        errors=errors,
    )


async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    # Runs in Starlette's outermost ServerErrorMiddleware, outside RequestIdMiddleware,
    # so the request id has to be logged and echoed here explicitly.
    request_id = getattr(request.state, "request_id", None)
    log.error("unhandled_error", exc_type=type(exc).__name__, request_id=request_id, exc_info=exc)
    response = problem_response(request, HTTPStatus.INTERNAL_SERVER_ERROR)
    if request_id:
        response.headers[REQUEST_ID_HEADER] = request_id
    return response


def register_error_handlers(app: FastAPI) -> None:
    """Attach all handlers; routes never build error responses themselves."""
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled_error)
