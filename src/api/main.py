"""FastAPI service for Melbourne auction clearance predictions.

    uvicorn src.api.main:app --reload
    open http://127.0.0.1:8000/docs

Design notes worth stating up front:

* **The model and context load once, at startup**, via the lifespan handler. A
  per-request ``joblib.load`` of a 3MB booster is the single most common way to
  make a fast model look slow.
* **Failure to load is not a crash.** The service starts in a degraded state and
  ``/health`` says so, so an orchestrator can route around it instead of
  crash-looping.
* **Unknown suburbs return 422, not a guess.** The model has no basis for
  scoring a suburb it has never seen, and a plausible-looking wrong number is
  worse than an error.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src import config
from src.api.context import MarketContext, load_context
from src.api.features import build_feature_row
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    Drivers,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PropertyRequest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_VERSION = "lgbm-1.0.0"

#: Thresholds for the coarse label. Chosen from the calibration table: the model
#: is well calibrated in the middle of its range, so a band either side of the
#: base rate is where "uncertain" genuinely means uncertain.
LIKELY_SELL = 0.75
LIKELY_PASS = 0.55

STATE: dict = {"bundle": None, "context": None}


def _load_artifacts() -> None:
    """Load the model bundle and market context; tolerate absence."""
    import joblib

    try:
        STATE["bundle"] = joblib.load(config.MODEL_PATH)
        log.info("loaded model %s (%d features, %d rounds)",
                 config.MODEL_PATH.name, len(STATE["bundle"]["features"]),
                 STATE["bundle"]["n_rounds"])
    except Exception as exc:  # noqa: BLE001 - startup must not hard-fail
        log.error("could not load model from %s: %s", config.MODEL_PATH, exc)

    try:
        STATE["context"] = load_context()
        log.info("loaded market context as of %s", STATE["context"].as_of.date())
    except Exception as exc:  # noqa: BLE001
        log.error("could not load market context: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield
    STATE.clear()


app = FastAPI(
    title="Melbourne Auction Clearance API",
    description=(
        "Calibrated probability that a Melbourne residential property sells at "
        "auction, from property attributes, spatial features (CBD and PTV train "
        "proximity), trailing suburb and region clearance rates, the RBA cash "
        "rate, and BOM weather."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def _require_ready() -> tuple[dict, MarketContext]:
    bundle, ctx = STATE.get("bundle"), STATE.get("context")
    if bundle is None or ctx is None:
        raise HTTPException(
            status_code=503,
            detail="model or market context unavailable; run `make train` first",
        )
    return bundle, ctx


def _band(p: float) -> tuple[str, str]:
    if p >= LIKELY_SELL:
        return "likely_sell", "high"
    if p <= LIKELY_PASS:
        return "likely_pass_in", "high" if p < 0.40 else "moderate"
    return "uncertain", "moderate"


def _predict_one(req: PropertyRequest, bundle: dict,
                 ctx: MarketContext) -> PredictionResponse:
    try:
        X, drivers, warnings = build_feature_row(req, ctx, bundle)
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown suburb {exc.args[0]!r}. The model covers "
                f"{len(ctx.known_suburbs())} Melbourne metropolitan suburbs; "
                f"see GET /suburbs for the list."
            ),
        ) from None

    p = float(bundle["booster"].predict(X)[0])
    outcome, confidence = _band(p)
    return PredictionResponse(
        clearance_probability=round(p, 4),
        predicted_outcome=outcome,
        confidence_band=confidence,
        drivers=Drivers(**drivers),
        context_as_of=ctx.as_of.date(),
        model_version=MODEL_VERSION,
        warnings=warnings,
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness and readiness in one call."""
    bundle, ctx = STATE.get("bundle"), STATE.get("context")
    ready = bundle is not None and ctx is not None
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=bundle is not None,
        context_loaded=ctx is not None,
        context_as_of=ctx.as_of.date() if ctx is not None else None,
        n_suburbs=len(ctx.known_suburbs()) if ctx is not None else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """What is actually deployed, and how it scored on the held-out year."""
    bundle, _ = _require_ready()
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        trained_through=bundle["trained_through"],
        train_rows=bundle["train_rows"],
        n_features=len(bundle["features"]),
        n_rounds=bundle["n_rounds"],
        test_metrics=bundle["test_metrics"],
        top_features=bundle["feature_importance"][:10],
    )


@app.get("/suburbs", tags=["reference"])
def suburbs() -> dict:
    """Suburbs the model can score."""
    _, ctx = _require_ready()
    return {"count": len(ctx.known_suburbs()), "suburbs": ctx.known_suburbs()}


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict(req: PropertyRequest) -> PredictionResponse:
    """Score a single property."""
    bundle, ctx = _require_ready()
    return _predict_one(req, bundle, ctx)


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
def predict_batch(req: BatchRequest) -> BatchResponse:
    """Score a whole campaign in one call.

    An agency uploading Saturday's book wants one request, not 80.
    """
    bundle, ctx = _require_ready()
    preds = [_predict_one(p, bundle, ctx) for p in req.properties]
    probs = np.array([p.clearance_probability for p in preds])
    return BatchResponse(
        predictions=preds,
        summary={
            "n": len(preds),
            "mean_clearance_probability": round(float(probs.mean()), 4),
            "expected_sales": round(float(probs.sum()), 1),
            "likely_sell": int(sum(p.predicted_outcome == "likely_sell" for p in preds)),
            "uncertain": int(sum(p.predicted_outcome == "uncertain" for p in preds)),
            "likely_pass_in": int(sum(p.predicted_outcome == "likely_pass_in" for p in preds)),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
