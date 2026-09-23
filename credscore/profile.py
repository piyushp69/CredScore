"""Human-friendly applicant profile and its mapping onto base model features.

Underwriters think in years, amounts and counts, not in Home Credit's column
conventions (negative day offsets, Y/N flags, per-table aggregates). The
profile is what the UI and API accept; `profile_to_base_features` translates it.

Base features the profile does not cover are left out of the returned dict so
`FeatureSpec.frame_from_records` fills them with training defaults. Features
that are explicitly unknown for this applicant (no bureau history, no car...)
are set to None, i.e. missing, which is how the training data encodes them.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .features import FEATURE_GROUPS

DAYS_PER_YEAR = 365.25
NOT_EMPLOYED_INCOME_TYPES = {"Pensioner", "Unemployed"}

# Profile fields validated against the model's category vocabulary.
CATEGORICAL_FIELDS = {
    "family_status": "NAME_FAMILY_STATUS",
    "education": "NAME_EDUCATION_TYPE",
    "housing_type": "NAME_HOUSING_TYPE",
    "income_type": "NAME_INCOME_TYPE",
    "occupation": "OCCUPATION_TYPE",
    "organization_type": "ORGANIZATION_TYPE",
    "contract_type": "NAME_CONTRACT_TYPE",
}


class ApplicantProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Personal
    age_years: float = Field(40, ge=18, le=100, description="Applicant age in years.")
    family_status: str = "Married"
    children: int = Field(0, ge=0, le=20)
    family_members: int = Field(2, ge=1, le=25, description="Household size including the applicant.")
    education: str = "Secondary / secondary special"
    housing_type: str = "House / apartment"
    owns_realty: bool = True
    owns_car: bool = False
    car_age_years: Optional[float] = Field(None, ge=0, le=100)
    region_rating: int = Field(2, ge=1, le=3, description="Home Credit region rating: 1 (best) to 3.")

    # Employment and income
    income_type: str = "Working"
    occupation: Optional[str] = None
    organization_type: Optional[str] = None
    annual_income: float = Field(..., gt=0)
    years_employed: Optional[float] = Field(
        None, ge=0, le=60, description="Years in current job; leave empty if not employed."
    )

    # Loan request
    contract_type: str = "Cash loans"
    credit_amount: float = Field(..., gt=0)
    annuity_amount: float = Field(..., gt=0, description="Periodic loan payment.")
    goods_price: Optional[float] = Field(None, gt=0, description="Price of goods financed; defaults to the credit amount.")

    # External credit scores, normalized to 0-1 (higher is safer)
    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)

    # Credit bureau history (other lenders)
    bureau_loans: int = Field(0, ge=0, le=500, description="Loans ever reported to the bureau; 0 means no history.")
    bureau_active_loans: int = Field(0, ge=0, le=500)
    bureau_total_credit: float = Field(0, ge=0)
    bureau_total_debt: float = Field(0, ge=0)
    bureau_overdue_amount: float = Field(0, ge=0, description="Largest amount ever overdue.")
    bureau_days_overdue: int = Field(0, ge=0, le=3000, description="Current days overdue on bureau loans.")
    bureau_years_since_last_loan: Optional[float] = Field(None, ge=0, le=50)
    bureau_enquiries_last_year: Optional[int] = Field(None, ge=0, le=100)

    # Previous applications with Home Credit
    prev_applications: int = Field(0, ge=0, le=200, description="0 means a new client.")
    prev_approved: int = Field(0, ge=0, le=200)
    prev_refused: int = Field(0, ge=0, le=200)

    # Repayment history on previous Home Credit loans
    installments_paid: int = Field(0, ge=0, le=2000, description="0 means no repayment history.")
    late_payment_share: float = Field(0, ge=0, le=1)
    avg_days_past_due: float = Field(0, ge=0, le=3000)
    max_days_past_due: float = Field(0, ge=0, le=3000)
    payment_ratio: float = Field(1.0, ge=0, le=5, description="Average share of each installment actually paid.")

    @model_validator(mode="after")
    def _check_consistency(self) -> "ApplicantProfile":
        if self.bureau_active_loans > self.bureau_loans:
            raise ValueError("bureau_active_loans cannot exceed bureau_loans")
        if self.prev_approved + self.prev_refused > self.prev_applications:
            raise ValueError("prev_approved + prev_refused cannot exceed prev_applications")
        if self.max_days_past_due < self.avg_days_past_due:
            raise ValueError("max_days_past_due cannot be below avg_days_past_due")
        return self


def _missing_group(group: str) -> dict:
    return {feature: None for feature in FEATURE_GROUPS[group]}


def profile_to_base_features(profile: ApplicantProfile) -> dict:
    p = profile
    not_employed = p.income_type in NOT_EMPLOYED_INCOME_TYPES
    features: dict = {
        "DAYS_BIRTH": -p.age_years * DAYS_PER_YEAR,
        "NAME_FAMILY_STATUS": p.family_status,
        "CNT_CHILDREN": p.children,
        "CNT_FAM_MEMBERS": p.family_members,
        "NAME_EDUCATION_TYPE": p.education,
        "NAME_HOUSING_TYPE": p.housing_type,
        "FLAG_OWN_REALTY": float(p.owns_realty),
        "FLAG_OWN_CAR": float(p.owns_car),
        "REGION_RATING_CLIENT": p.region_rating,
        "REGION_RATING_CLIENT_W_CITY": p.region_rating,
        "NAME_INCOME_TYPE": p.income_type,
        "OCCUPATION_TYPE": p.occupation,
        "AMT_INCOME_TOTAL": p.annual_income,
        "NAME_CONTRACT_TYPE": p.contract_type,
        "AMT_CREDIT": p.credit_amount,
        "AMT_ANNUITY": p.annuity_amount,
        "AMT_GOODS_PRICE": p.goods_price if p.goods_price is not None else p.credit_amount,
        "EXT_SOURCE_1": p.ext_source_1,
        "EXT_SOURCE_2": p.ext_source_2,
        "EXT_SOURCE_3": p.ext_source_3,
    }
    # Unknown values fall back to training defaults (by omission) unless the
    # data encodes them as genuinely missing.
    if not p.owns_car:
        features["OWN_CAR_AGE"] = None
    elif p.car_age_years is not None:
        features["OWN_CAR_AGE"] = p.car_age_years

    if p.years_employed is not None:
        features["DAYS_EMPLOYED"] = -p.years_employed * DAYS_PER_YEAR
    elif not_employed:
        features["DAYS_EMPLOYED"] = None

    if p.organization_type is not None:
        features["ORGANIZATION_TYPE"] = p.organization_type
    elif not_employed:
        features["ORGANIZATION_TYPE"] = "XNA"

    if p.bureau_enquiries_last_year is not None:
        features["AMT_REQ_CREDIT_BUREAU_YEAR"] = p.bureau_enquiries_last_year

    if p.bureau_loans == 0:
        features |= _missing_group("bureau")
    else:
        features |= {
            "BUREAU_LOAN_COUNT": p.bureau_loans,
            "BUREAU_ACTIVE_COUNT": p.bureau_active_loans,
            "BUREAU_AMT_CREDIT_SUM_DEBT": p.bureau_total_debt,
            "BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX": p.bureau_overdue_amount,
            "BUREAU_CREDIT_DAY_OVERDUE_MAX": p.bureau_days_overdue,
        }
        # A reported loan always has a credit amount, so 0 here means "not supplied"
        # and the training default is a better guess than a contradictory zero.
        if p.bureau_total_credit:
            features["BUREAU_AMT_CREDIT_SUM"] = p.bureau_total_credit
        if p.bureau_years_since_last_loan is not None:
            features["BUREAU_DAYS_CREDIT_MAX"] = -p.bureau_years_since_last_loan * DAYS_PER_YEAR

    if p.prev_applications == 0:
        features |= _missing_group("previous")
    else:
        features |= {
            "PREV_APP_COUNT": p.prev_applications,
            "PREV_APPROVED_COUNT": p.prev_approved,
            "PREV_REFUSED_COUNT": p.prev_refused,
        }

    if p.installments_paid == 0:
        features |= _missing_group("installments")
    else:
        features |= {
            "INSTAL_COUNT": p.installments_paid,
            "INSTAL_LATE_SHARE": p.late_payment_share,
            "INSTAL_DPD_MEAN": p.avg_days_past_due,
            "INSTAL_DPD_MAX": p.max_days_past_due,
            "INSTAL_PAYMENT_RATIO_MEAN": p.payment_ratio,
        }
    return features


def profile_defaults(spec_defaults: dict) -> dict:
    """Profile field defaults that describe a typical training applicant."""

    def median(feature: str, fallback: float) -> float:
        value = spec_defaults.get(feature)
        return fallback if value is None else float(value)

    def mode(feature: str, fallback: str) -> str:
        value = spec_defaults.get(feature)
        return fallback if value is None else str(value)

    defaults = {name: f.default for name, f in ApplicantProfile.model_fields.items() if not f.is_required()}
    defaults.update(
        age_years=round(-median("DAYS_BIRTH", -40 * DAYS_PER_YEAR) / DAYS_PER_YEAR, 1),
        family_status=mode("NAME_FAMILY_STATUS", "Married"),
        education=mode("NAME_EDUCATION_TYPE", "Secondary / secondary special"),
        housing_type=mode("NAME_HOUSING_TYPE", "House / apartment"),
        income_type=mode("NAME_INCOME_TYPE", "Working"),
        contract_type=mode("NAME_CONTRACT_TYPE", "Cash loans"),
        annual_income=round(median("AMT_INCOME_TOTAL", 150_000)),
        credit_amount=round(median("AMT_CREDIT", 500_000)),
        annuity_amount=round(median("AMT_ANNUITY", 25_000)),
        goods_price=round(median("AMT_GOODS_PRICE", 450_000)),
        years_employed=round(-median("DAYS_EMPLOYED", -3 * DAYS_PER_YEAR) / DAYS_PER_YEAR, 1),
        ext_source_2=round(median("EXT_SOURCE_2", 0.5), 3),
        ext_source_3=round(median("EXT_SOURCE_3", 0.5), 3),
    )
    return defaults
