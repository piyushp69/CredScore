"""Model evaluation: discrimination, calibration, business and fairness views."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score, roc_curve

from .scoring import APPROVE, DECLINE, REVIEW, RISK_BANDS, DecisionPolicy, pd_to_score


def _r(value, digits: int = 4):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else round(float(value), digits)


def ranking_metrics(y, s) -> dict:
    """Metrics that only need a risk ordering (higher score = riskier)."""
    y, s = np.asarray(y), np.asarray(s)
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    return {
        "roc_auc": _r(auc),
        "gini": _r(2 * auc - 1),
        "pr_auc": _r(average_precision_score(y, s)),
        "ks": _r(np.max(tpr - fpr)),
        "n": int(len(y)),
    }


def classification_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return ranking_metrics(y, p) | {
        "brier": _r(brier_score_loss(y, p), 5),
        "log_loss": _r(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), 5),
        "base_rate": _r(y.mean()),
        "mean_pd": _r(p.mean()),
        "n": int(len(y)),
    }


def _downsample(*arrays, n_points: int = 150):
    idx = np.unique(np.linspace(0, len(arrays[0]) - 1, n_points).astype(int))
    return [np.asarray(a)[idx] for a in arrays]


def roc_points(y, p, n_points: int = 150) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    fpr, tpr = _downsample(fpr, tpr, n_points=n_points)
    return {"fpr": [_r(v) for v in fpr], "tpr": [_r(v) for v in tpr]}


def pr_points(y, p, n_points: int = 150) -> dict:
    precision, recall, _ = precision_recall_curve(y, p)
    precision, recall = _downsample(precision, recall, n_points=n_points)
    return {"precision": [_r(v) for v in precision], "recall": [_r(v) for v in recall]}


def calibration_table(y, p, n_bins: int = 10) -> list[dict]:
    """Observed default rate vs mean predicted PD, in equal-count bins."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    table = df.groupby("bin").agg(mean_pd=("p", "mean"), default_rate=("y", "mean"), n=("y", "size"))
    return [
        {"bin": int(b) + 1, "mean_pd": _r(r.mean_pd), "default_rate": _r(r.default_rate), "n": int(r.n)}
        for b, r in table.iterrows()
    ]


def gains_table(y, p, n_bins: int = 10) -> list[dict]:
    """Risk deciles, riskiest first: default rate, lift and cumulative share of defaulters caught."""
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = np.arange(len(df)) * n_bins // len(df) + 1
    base_rate, total_bad = df["y"].mean(), df["y"].sum()
    rows, cum_bad = [], 0
    for decile, g in df.groupby("decile"):
        cum_bad += g["y"].sum()
        rows.append({
            "decile": int(decile),
            "n": int(len(g)),
            "min_pd": _r(g["p"].min()),
            "max_pd": _r(g["p"].max()),
            "default_rate": _r(g["y"].mean()),
            "lift": _r(g["y"].mean() / base_rate if base_rate else np.nan, 3),
            "cum_population": _r(decile / n_bins),
            "cum_defaulters_captured": _r(cum_bad / total_bad if total_bad else np.nan),
        })
    return rows


def band_table(y, p) -> list[dict]:
    scores = pd_to_score(p)
    y = np.asarray(y)
    rows = []
    for band in RISK_BANDS:
        mask = (scores >= band.min_score) & (scores <= band.max_score)
        rows.append({
            "band": band.code,
            "label": band.label,
            "min_score": band.min_score,
            "max_score": band.max_score,
            "n": int(mask.sum()),
            "share": _r(mask.mean()),
            "default_rate": _r(y[mask].mean()) if mask.any() else None,
        })
    return rows


def policy_table(y, p, policy: DecisionPolicy) -> list[dict]:
    y, p = np.asarray(y), np.asarray(p)
    decisions = np.array([policy.decide(v) for v in p])
    total_bad = y.sum()
    rows = []
    for decision in (APPROVE, REVIEW, DECLINE):
        mask = decisions == decision
        rows.append({
            "decision": decision,
            "n": int(mask.sum()),
            "share": _r(mask.mean()),
            "default_rate": _r(y[mask].mean()) if mask.any() else None,
            "share_of_defaulters": _r(y[mask].sum() / total_bad) if total_bad else None,
        })
    return rows


def score_histogram(y, p, bin_width: int = 20) -> dict:
    scores = pd_to_score(p)
    edges = np.arange(300, 850 + bin_width, bin_width)
    y = np.asarray(y)
    good, _ = np.histogram(scores[y == 0], bins=edges)
    bad, _ = np.histogram(scores[y == 1], bins=edges)
    return {
        "bin_start": edges[:-1].tolist(),
        "bin_width": bin_width,
        "repaid": (good / max(good.sum(), 1)).round(5).tolist(),
        "defaulted": (bad / max(bad.sum(), 1)).round(5).tolist(),
    }


def group_report(y, p, groups: pd.Series, policy: DecisionPolicy, min_size: int = 200) -> list[dict]:
    """Per-group outcomes, for fairness monitoring."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": np.asarray(groups, dtype=object)})
    df = df.dropna(subset=["g"])
    if isinstance(groups.dtype, pd.CategoricalDtype):
        order = [c for c in groups.cat.categories if c in set(df["g"])]
    else:
        order = sorted(df["g"].unique(), key=str)
    rows = []
    for group in order:
        g = df[df["g"] == group]
        if len(g) < min_size:
            continue
        approved = g["p"] < policy.approve_below
        declined = g["p"] >= policy.decline_at
        rows.append({
            "group": str(group),
            "n": int(len(g)),
            "default_rate": _r(g["y"].mean()),
            "mean_pd": _r(g["p"].mean()),
            "approval_rate": _r(approved.mean()),
            "decline_rate": _r(declined.mean()),
            "roc_auc": _r(roc_auc_score(g["y"], g["p"])) if g["y"].nunique() == 2 else None,
        })
    return rows


def age_band(days_birth: pd.Series) -> pd.Series:
    age = -pd.to_numeric(days_birth, errors="coerce") / 365.25
    return pd.cut(age, bins=[0, 30, 40, 50, 60, 200], labels=["Under 30", "30-39", "40-49", "50-59", "60+"], right=False)
