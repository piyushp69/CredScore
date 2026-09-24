"""Shared pieces of the Streamlit dashboard: model loading, formatting, charts, presets."""

from __future__ import annotations

import random

import plotly.graph_objects as go
import streamlit as st

from credscore.model import ModelNotFoundError
from credscore.service import ScoringService

# Colour-vision-safe palette (Okabe-Ito based), shared by every chart.
BLUE = "#4c9be8"
ORANGE = "#e69f00"
NEUTRAL = "#8b93a7"
MUTED = "#6b7280"
RISK_UP = "#e5484d"
RISK_DOWN = "#30a46c"

DECISION = {
    "APPROVE": {"label": "Approve", "icon": "✅", "color": "#30a46c"},
    "REVIEW": {"label": "Manual review", "icon": "⚠️", "color": "#f5a623"},
    "DECLINE": {"label": "Decline", "icon": "⛔", "color": "#e5484d"},
}
BAND_COLOR = {"good": "#30a46c", "warning": "#f5a623", "serious": "#f76b15", "critical": "#e5484d"}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the model…")
def _load_service() -> ScoringService:
    return ScoringService.load()


def load_service() -> tuple[ScoringService | None, str | None]:
    """The scoring service, or the reason it could not be loaded. Failures are not cached."""
    try:
        return _load_service(), None
    except ModelNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # corrupt bundle, bad policy override...
        return None, f"Could not load the model bundle: {exc}"


def require_service() -> ScoringService:
    """The loaded service; otherwise shows an error and stops the page."""
    service, error = load_service()
    if service is None:
        st.error(f"**No model loaded.** {error}\n\nTrain one with `python -m credscore.pipeline all`.")
        st.stop()
    return service


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def pct(v, digits: int = 1) -> str:
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def num(v, digits: int = 0) -> str:
    return "—" if v is None else f"{v:,.{digits}f}"


def dec(v, digits: int = 3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def targets_note(info: dict) -> str:
    """How training chose the decision cut-offs, from the approval targets stored in the bundle."""
    targets = info.get("policy_targets")
    if not targets:
        return ""
    return (f"Cut-offs were set on the validation set to approve about {pct(targets['approve_rate'], 0)} "
            f"and decline the riskiest {pct(targets['decline_rate'], 0)}.")


def page_header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def figure(height: int = 360, **layout) -> go.Figure:
    """A Plotly figure with the dashboard's shared styling; transparent so it follows the theme."""
    fig = go.Figure()
    fig.update_layout({
        "height": height,
        "margin": dict(l=8, r=16, t=36, b=8),
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        "hoverlabel": dict(namelength=-1),
        **layout,
    })
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def diagonal(fig: go.Figure, upto: float = 1.0, name: str = "Random") -> None:
    fig.add_trace(go.Scatter(x=[0, upto], y=[0, upto], mode="lines", name=name, hoverinfo="skip",
                             line=dict(color=MUTED, dash="dash", width=1)))


# --------------------------------------------------------------------------- #
# Example applicants
# --------------------------------------------------------------------------- #
STRONG = dict(
    age_years=45, family_status="Married", children=1, family_members=3, education="Higher education",
    housing_type="House / apartment", owns_realty=True, owns_car=True, car_age_years=5, region_rating=1,
    income_type="State servant", occupation="Core staff", organization_type=None, annual_income=270000,
    years_employed=12, contract_type="Cash loans", credit_amount=450000, annuity_amount=22000,
    goods_price=450000, ext_source_1=0.72, ext_source_2=0.74, ext_source_3=0.7, bureau_loans=6,
    bureau_active_loans=1, bureau_total_credit=1200000, bureau_total_debt=90000, bureau_overdue_amount=0,
    bureau_days_overdue=0, bureau_years_since_last_loan=2, bureau_enquiries_last_year=1, prev_applications=4,
    prev_approved=4, prev_refused=0, installments_paid=60, late_payment_share=0, avg_days_past_due=0,
    max_days_past_due=0, payment_ratio=1,
)

RISKY = dict(
    age_years=24, family_status="Single / not married", children=0, family_members=1,
    education="Secondary / secondary special", housing_type="With parents", owns_realty=False, owns_car=False,
    car_age_years=None, region_rating=3, income_type="Working", occupation="Laborers", organization_type=None,
    annual_income=90000, years_employed=0.5, contract_type="Cash loans", credit_amount=900000,
    annuity_amount=45000, goods_price=810000, ext_source_1=None, ext_source_2=0.15, ext_source_3=0.2,
    bureau_loans=6, bureau_active_loans=5, bureau_total_credit=600000, bureau_total_debt=450000,
    bureau_overdue_amount=25000, bureau_days_overdue=30, bureau_years_since_last_loan=0.2,
    bureau_enquiries_last_year=6, prev_applications=5, prev_approved=2, prev_refused=3, installments_paid=20,
    late_payment_share=0.3, avg_days_past_due=6, max_days_past_due=40, payment_ratio=0.9,
)


def demo_batch(typical: dict, n: int = 250, seed: int = 7) -> list[dict]:
    """Deterministic pseudo-random variations of the presets, for trying batch scoring."""
    rng = random.Random(seed)

    def jitter(value: float, spread: float) -> float:
        return value * (1 + (rng.random() - 0.5) * spread)

    def clamp01(v: float) -> float:
        return round(min(0.95, max(0.01, v)), 3)

    anchors = [typical, STRONG, RISKY]
    rows = []
    for i in range(n):
        row = dict(anchors[0 if rng.random() < 0.6 else 1 if rng.random() < 0.62 else 2])
        credit = max(50000, round(jitter(row["credit_amount"], 1.1) / 1000) * 1000)
        rows.append({
            **row,
            "applicant_id": f"DEMO-{i + 1:04d}",
            "age_years": min(68, max(21, round(jitter(row["age_years"], 0.45)))),
            "annual_income": max(30000, round(jitter(row["annual_income"], 0.9) / 1000) * 1000),
            "credit_amount": credit,
            "goods_price": round(credit * (0.85 + rng.random() * 0.15) / 1000) * 1000,
            "annuity_amount": round(credit * (0.03 + rng.random() * 0.05)),
            "ext_source_2": None if row["ext_source_2"] is None else clamp01(row["ext_source_2"] + (rng.random() - 0.5) * 0.3),
            "ext_source_3": None if row["ext_source_3"] is None else clamp01(row["ext_source_3"] + (rng.random() - 0.5) * 0.3),
        })
    return rows

