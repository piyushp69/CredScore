"""Request / response models for the public API (also drive the OpenAPI docs)."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

MAX_BATCH_SIZE = 10_000

Decision = Literal["APPROVE", "REVIEW", "DECLINE"]
FeatureValue = Union[float, str, None]

EXAMPLE_PROFILE = {
    "age_years": 38,
    "family_status": "Married",
    "children": 1,
    "family_members": 3,
    "education": "Higher education",
    "housing_type": "House / apartment",
    "owns_realty": True,
    "owns_car": True,
    "car_age_years": 6,
    "income_type": "Working",
    "occupation": "Core staff",
    "annual_income": 225000,
    "years_employed": 7.5,
    "contract_type": "Cash loans",
    "credit_amount": 600000,
    "annuity_amount": 28000,
    "goods_price": 540000,
    "ext_source_2": 0.68,
    "ext_source_3": 0.62,
    "bureau_loans": 4,
    "bureau_active_loans": 1,
    "bureau_total_credit": 850000,
    "bureau_total_debt": 120000,
    "prev_applications": 2,
    "prev_approved": 2,
    "installments_paid": 24,
    "late_payment_share": 0.0,
}


class RiskBand(BaseModel):
    code: str
    label: str
    min_score: int
    max_score: int
    status: Literal["good", "warning", "serious", "critical"]


class FeatureView(BaseModel):
    label: str
    value: FeatureValue
    display_value: str


class Explanation(FeatureView):
    feature: str
    contribution: float = Field(description="SHAP contribution in log-odds; positive raises the PD.")
    effect: Literal["increases_risk", "decreases_risk"]
    source: Literal["applicant", "default", "derived"] = Field(
        description="applicant = supplied in the request, default = training default, derived = computed."
    )


class ScoreResult(BaseModel):
    probability_of_default: float
    credit_score: int
    risk_band: RiskBand
    decision: Decision
    decision_reason: str
    model_version: str
    key_metrics: dict[str, FeatureView] = {}
    explanations: list[Explanation] = []
    reasons: list[str] = Field(default=[], description="Main factors raising the risk, in plain language.")
    base_log_odds: Optional[float] = None
    warnings: list[str] = []


class FeatureScoreRequest(BaseModel):
    features: dict[str, FeatureValue] = Field(
        description="Base model features by name. Omitted features take training defaults; null means missing."
    )


class BatchRequest(BaseModel):
    applicants: list[dict[str, Any]] = Field(
        min_length=1, max_length=MAX_BATCH_SIZE,
        description="ApplicantProfile objects; an optional 'applicant_id' key is echoed back.",
    )


class BatchItem(BaseModel):
    index: int
    applicant_id: Optional[str] = None
    result: Optional[ScoreResult] = None
    error: Optional[str] = None


class BatchSummary(BaseModel):
    submitted: int
    scored: int
    failed: int
    decisions: dict[str, int]
    mean_probability_of_default: Optional[float]
    mean_credit_score: Optional[float]


class BatchResponse(BaseModel):
    summary: BatchSummary
    results: list[BatchItem]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: Optional[str] = None
    detail: Optional[str] = None
