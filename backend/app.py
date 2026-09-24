"""CredScore REST API.

Run with:  uvicorn backend.app:app --reload --port 8000
Docs at:   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from credscore import __version__
from credscore.model import ModelNotFoundError
from credscore.profile import ApplicantProfile
from credscore.service import ScoringService

from .schemas import (
    EXAMPLE_PROFILE,
    BatchRequest,
    BatchResponse,
    FeatureScoreRequest,
    Health,
    ScoreResult,
)

log = logging.getLogger("credscore.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service, app.state.load_error = None, None
    try:
        app.state.service = ScoringService.load()
        log.info("Loaded model %s", app.state.service.model.version)
    except ModelNotFoundError as exc:
        app.state.load_error = str(exc)
        log.warning("API started without a model: %s", exc)
    except Exception as exc:  # corrupt bundle, bad policy override...: serve /health, not a crash loop
        app.state.load_error = f"Could not load the model bundle: {exc}"
        log.exception("API started without a model")
    yield


def get_service(request: Request) -> ScoringService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail=request.app.state.load_error or "Model not loaded")
    return service


Service = Annotated[ScoringService, Depends(get_service)]
router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=Health, tags=["meta"])
def health(request: Request):
    service = request.app.state.service
    if service is None:
        return Health(status="degraded", model_loaded=False, detail=request.app.state.load_error)
    return Health(status="ok", model_loaded=True, model_version=service.model.version)


@router.get("/model", tags=["model"])
def model_info(service: Service):
    """Model version, test-set metrics, decision policy and scorecard definition."""
    return service.model_info()


@router.get("/model/performance", tags=["model"])
def model_performance(service: Service):
    """Evaluation report: curves, calibration, gains, bands, policy outcomes, fairness."""
    report = service.performance_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No evaluation report found; retrain the model.")
    return report


@router.get("/model/importance", tags=["model"])
def model_importance(service: Service, top: int = Query(20, ge=1, le=100)):
    report = service.performance_report() or {}
    return {"importance": report.get("importance", [])[:top]}


@router.get("/schema", tags=["model"])
def schema(service: Service):
    """Applicant profile fields, allowed category values, typical defaults and base features."""
    return service.schema()


@router.get("/insights", tags=["portfolio"])
def insights(service: Service):
    """Observed default rates across applicant segments of the training portfolio."""
    data = service.insights()
    if data is None:
        raise HTTPException(status_code=404, detail="No insights found; run `python -m credscore.pipeline insights`.")
    return data


@router.post("/score", response_model=ScoreResult, tags=["scoring"])
def score(
    service: Service,
    profile: Annotated[ApplicantProfile, Body(openapi_examples={"applicant": {"value": EXAMPLE_PROFILE}})],
    top_k: int = Query(10, ge=0, le=50, description="Number of feature explanations to return."),
):
    """Score one applicant: PD, credit score, risk band, decision and explanations."""
    errors = service.profile_errors(profile)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return service.score_profiles([profile], top_k=top_k)[0]


@router.post("/score/features", response_model=ScoreResult, tags=["scoring"])
def score_features(service: Service, request: FeatureScoreRequest, top_k: int = Query(10, ge=0, le=50)):
    """Score raw base features (advanced). Unknown feature names are ignored with a warning."""
    unknown = service.unknown_features(request.features)
    result = service.score_records([request.features], top_k=top_k)[0]
    if unknown:
        result["warnings"] = [f"Ignored unknown features: {', '.join(unknown)}"]
    return result


@router.post("/score/batch", response_model=BatchResponse, tags=["scoring"])
def score_batch(
    service: Service,
    request: BatchRequest,
    reasons: int = Query(3, ge=0, le=10, description="Risk reasons per applicant (0 is fastest)."),
):
    """Score up to 10,000 applicant profiles. Invalid rows are reported, not fatal."""
    return service.score_batch(request.applicants, n_reasons=reasons)


def create_app() -> FastAPI:
    app = FastAPI(
        title="CredScore API",
        version=__version__,
        description="Credit default risk scoring with explanations, trained on the Home Credit dataset.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CREDSCORE_CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": "CredScore API", "version": __version__, "docs": "/docs", "health": "/api/v1/health"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000)
