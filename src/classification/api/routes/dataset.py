"""GET /v1/dataset — paginated LEDGAR rows (10 per page by default); GET /v1/labels."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query, Request

from ..errors import APIError
from ..schemas import DatasetPage, ErrorResponse, LabelList
from ..security import require_api_key

router = APIRouter(prefix="/v1", tags=["dataset"], dependencies=[Depends(require_api_key)])

Split = Literal["train", "validation", "test"]


@router.get("/dataset", response_model=DatasetPage, responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def get_dataset(
    request: Request,
    split: Split = Query("train", description="train | validation | test"),
    offset: int = Query(0, ge=0, description="Row offset"),
    limit: int = Query(10, ge=1, le=100, description="Rows per page (default 10, max 100)"),
) -> DatasetPage:
    """Sync handler -> runs in FastAPI's threadpool (first call may download ~28 MB)."""
    try:
        return DatasetPage(**request.app.state.dataset.page(split, offset, limit))
    except (OSError, ValueError) as exc:
        raise APIError(503, f"Dataset unavailable: {exc}", None, "dataset_unavailable") from exc


@router.get("/labels", response_model=LabelList)
def get_labels(request: Request) -> LabelList:
    """The 100 clause types the model can predict."""
    store = request.app.state.models
    latest = store.latest()
    predictor = store.loaded().get(latest.id) if latest else None
    names = predictor.label_names if predictor else request.app.state.dataset.labels()
    return LabelList(total=len(names), data=names)
