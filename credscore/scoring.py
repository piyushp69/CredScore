"""Turn a probability of default (PD) into a credit score, risk band and decision.

The score uses the standard "points to double the odds" scorecard scaling:

    score = offset + factor * ln(odds_good)       odds_good = (1 - PD) / PD
    factor = PDO / ln(2)
    offset = BASE_SCORE - factor * ln(BASE_ODDS)

With the defaults below, 650 points means 20:1 good:bad odds (PD ~4.8%) and
every 40 points doubles the odds.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

BASE_SCORE = 650
BASE_ODDS = 20
PDO = 40
MIN_SCORE = 300
MAX_SCORE = 850

_FACTOR = PDO / math.log(2)
_OFFSET = BASE_SCORE - _FACTOR * math.log(BASE_ODDS)
_EPS = 1e-6


@dataclass(frozen=True)
class RiskBand:
    code: str
    label: str
    min_score: int
    max_score: int
    status: str  # good | warning | serious | critical


# Ordered from safest to riskiest. Score ranges are inclusive.
RISK_BANDS = (
    RiskBand("A", "Very low risk", 720, MAX_SCORE, "good"),
    RiskBand("B", "Low risk", 660, 719, "good"),
    RiskBand("C", "Moderate risk", 600, 659, "warning"),
    RiskBand("D", "High risk", 540, 599, "serious"),
    RiskBand("E", "Very high risk", MIN_SCORE, 539, "critical"),
)

APPROVE, REVIEW, DECLINE = "APPROVE", "REVIEW", "DECLINE"


def pd_to_score(pd_values):
    """Map PD (scalar or array) to an integer score clipped to [MIN_SCORE, MAX_SCORE]."""
    p = np.clip(np.asarray(pd_values, dtype=float), _EPS, 1 - _EPS)
    score = np.clip(np.rint(_OFFSET + _FACTOR * np.log((1 - p) / p)), MIN_SCORE, MAX_SCORE).astype(int)
    return int(score) if score.ndim == 0 else score


def score_to_pd(score: float) -> float:
    """Inverse of `pd_to_score` (ignoring clipping and rounding)."""
    odds = math.exp((score - _OFFSET) / _FACTOR)
    return 1 / (1 + odds)


def risk_band(score: int) -> RiskBand:
    for band in RISK_BANDS:
        if score >= band.min_score:
            return band
    return RISK_BANDS[-1]


@dataclass(frozen=True)
class DecisionPolicy:
    """PD cut-offs: approve below `approve_below`, decline at or above `decline_at`."""

    approve_below: float
    decline_at: float

    def __post_init__(self):
        if not 0 < self.approve_below <= self.decline_at < 1:
            raise ValueError(
                f"Invalid policy: need 0 < approve_below ({self.approve_below}) "
                f"<= decline_at ({self.decline_at}) < 1"
            )

    def decide(self, pd_value: float) -> str:
        if pd_value < self.approve_below:
            return APPROVE
        if pd_value >= self.decline_at:
            return DECLINE
        return REVIEW

    def explain(self, pd_value: float) -> str:
        decision = self.decide(pd_value)
        if decision == APPROVE:
            return f"PD {pd_value:.1%} is below the approval cut-off of {self.approve_below:.1%}."
        if decision == DECLINE:
            return f"PD {pd_value:.1%} is at or above the decline cut-off of {self.decline_at:.1%}."
        return (
            f"PD {pd_value:.1%} falls between the approval ({self.approve_below:.1%}) "
            f"and decline ({self.decline_at:.1%}) cut-offs, so it needs manual review."
        )


def scorecard_description() -> dict:
    return {
        "base_score": BASE_SCORE,
        "base_odds": BASE_ODDS,
        "points_to_double_odds": PDO,
        "min_score": MIN_SCORE,
        "max_score": MAX_SCORE,
        "bands": [asdict(b) | {"max_pd": round(score_to_pd(b.min_score), 4)} for b in RISK_BANDS],
    }
