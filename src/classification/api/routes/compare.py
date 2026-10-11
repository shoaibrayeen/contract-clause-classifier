"""POST /v1/compare: the transformer and the classical model on the same provisions, side by side.

The input is split into provisions once, then every requested model scores the same list, so clause indexes
line up across models. Default models: the latest transformer and the latest classical model.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from ...registry import ModelNotFoundError
from ..errors import APIError
from ..schemas import CompareAgreement, CompareModelResult, CompareRequest, CompareResponse, ErrorResponse
from ..security import require_api_key

router = APIRouter(prefix="/v1", tags=["compare"], dependencies=[Depends(require_api_key)])
ERRORS = {400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
          422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


@router.post("/compare", response_model=CompareResponse, responses=ERRORS)
def compare(body: CompareRequest, request: Request) -> CompareResponse:
    s = request.app.state
    store = s.models
    text = body.text
    if text is None and body.messages:
        users = [m for m in body.messages if m.role == "user"]
        text = users[-1].text() if users else None
    if not text or not text.strip():
        raise APIError(400, "Provide the contract text as `text` or as the last user message", "text", "missing_text")
    if len(text) > s.settings.max_input_chars:
        raise APIError(413, f"Input exceeds {s.settings.max_input_chars} characters", "text", "input_too_large")

    if body.models:
        try:
            infos = [store.resolve(m) for m in body.models]
        except ModelNotFoundError as exc:
            raise APIError(404, str(exc), "models", "model_not_found") from exc
    else:
        wanted = {fam: store.latest_of(fam) for fam in ("transformer", "classical")}
        missing = [fam for fam, info in wanted.items() if info is None]
        if missing:
            hint = {"transformer": "`make all`", "classical": "`make classical`"}
            raise APIError(503, "No trained " + " or ".join(missing) + " model to compare with. Train with "
                           + " and ".join(hint[m] for m in missing) + ".", "models", "model_not_loaded")
        infos = list(wanted.values())

    loaded = []
    for info in infos:
        try:
            loaded.append(store.get(info.id))
        except Exception as exc:
            raise APIError(503, f"Model {info.id} could not be loaded: {exc}", "models", "model_not_loaded") from exc

    segment = True if body.segment is None else body.segment
    provisions = loaded[0][1].split(text, segment)  # one split, shared by every model
    results = []
    for info, predictor in loaded:
        t = time.perf_counter()
        out = predictor.predict_provisions_document(provisions, top_k=body.top_k, threshold=body.threshold)
        results.append(CompareModelResult(id=info.id, family=info.family, version=info.version,
                                          latency_ms=round((time.perf_counter() - t) * 1000, 1),
                                          document=out["document"], clauses=out["clauses"]))
    agreement = None
    if len(results) >= 2 and provisions:
        tops = [[c.predictions[0].name for c in r.clauses] for r in results]
        per = [len({t[i] for t in tops}) == 1 for i in range(len(provisions))]
        agreement = CompareAgreement(clauses_agree=sum(per), clauses_total=len(per),
                                     rate=round(sum(per) / len(per), 4), per_clause=per)
    return CompareResponse(num_clauses=len(provisions), models=results, agreement=agreement)
