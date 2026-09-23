import numpy as np
import pytest

from credscore import evaluation
from credscore.scoring import (
    APPROVE,
    BASE_SCORE,
    DECLINE,
    MAX_SCORE,
    MIN_SCORE,
    PDO,
    REVIEW,
    RISK_BANDS,
    DecisionPolicy,
    pd_to_score,
    risk_band,
    score_to_pd,
)


def test_score_scaling_anchors():
    assert pd_to_score(1 / 21) == BASE_SCORE  # 20:1 odds
    odds_doubled = 1 / (1 + 40)
    assert pd_to_score(odds_doubled) == BASE_SCORE + PDO
    assert score_to_pd(BASE_SCORE) == pytest.approx(1 / 21)


def test_score_is_monotonic_and_clipped():
    pds = np.linspace(0.0, 1.0, 101)
    scores = pd_to_score(pds)
    assert np.all(np.diff(scores) <= 0)
    assert scores.max() == MAX_SCORE and scores.min() == MIN_SCORE
    assert isinstance(pd_to_score(0.1), int)


def test_bands_cover_the_scale_without_gaps():
    ordered = sorted(RISK_BANDS, key=lambda b: b.min_score)
    assert ordered[0].min_score == MIN_SCORE and ordered[-1].max_score == MAX_SCORE
    for lower, upper in zip(ordered, ordered[1:]):
        assert upper.min_score == lower.max_score + 1
    assert risk_band(850).code == "A" and risk_band(300).code == "E" and risk_band(659).code == "C"


def test_decision_policy():
    policy = DecisionPolicy(approve_below=0.08, decline_at=0.2)
    assert policy.decide(0.05) == APPROVE
    assert policy.decide(0.08) == REVIEW
    assert policy.decide(0.2) == DECLINE
    assert "manual review" in policy.explain(0.1)
    with pytest.raises(ValueError):
        DecisionPolicy(approve_below=0.3, decline_at=0.2)


def test_evaluation_tables():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.6, 5000)
    y = (rng.random(5000) < p).astype(int)
    metrics = evaluation.classification_metrics(y, p)
    assert 0.6 < metrics["roc_auc"] < 0.9 and metrics["gini"] == pytest.approx(2 * metrics["roc_auc"] - 1, abs=1e-3)

    calibration = evaluation.calibration_table(y, p)
    assert len(calibration) == 10 and sum(r["n"] for r in calibration) == 5000
    assert all(abs(r["mean_pd"] - r["default_rate"]) < 0.08 for r in calibration)

    gains = evaluation.gains_table(y, p)
    assert gains[-1]["cum_defaulters_captured"] == pytest.approx(1.0)
    assert gains[0]["default_rate"] > gains[-1]["default_rate"]

    policy = DecisionPolicy(0.1, 0.4)
    outcomes = evaluation.policy_table(y, p, policy)
    assert sum(r["share"] for r in outcomes) == pytest.approx(1.0, abs=1e-3)
    bands = evaluation.band_table(y, p)
    assert sum(r["n"] for r in bands) == 5000
