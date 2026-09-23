"""Portfolio insights: observed default rates across applicant segments.

Computed from the full labelled feature table, so the dashboard shows real
portfolio behaviour instead of placeholder numbers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..evaluation import age_band
from ..features import TARGET_COLUMN, add_derived_features

log = logging.getLogger(__name__)
INF = np.inf


def _bins(values: pd.Series, edges: list[float], labels: list[str], missing: str | None = None) -> pd.Series:
    """Ordered bins; rows with no value land in the `missing` bin, shown first."""
    out = pd.cut(values, bins=edges, labels=labels, right=False)
    if missing:
        out = out.cat.add_categories([missing]).fillna(missing).cat.reorder_categories([missing, *labels])
    return out


def _ordered_labels(groups) -> list | None:
    dtype = getattr(groups, "dtype", None)
    return list(dtype.categories) if isinstance(dtype, pd.CategoricalDtype) else None


def _segment(key, title, groups, y, ordered, description, order=None, top=None):
    # Ordered breakdowns keep their natural bin order; nominal ones are sorted by size.
    order = order or (_ordered_labels(groups) if ordered else None)
    df = pd.DataFrame({"segment": pd.Series(groups, index=y.index).astype(object), "y": y}).dropna(subset=["segment"])
    stats = df.groupby("segment").agg(applicants=("y", "size"), default_rate=("y", "mean"))
    if order is not None:
        stats = stats.reindex([o for o in order if o in stats.index])
    else:
        stats = stats.sort_values("applicants", ascending=False)
    if top and len(stats) > top:
        head, tail = stats.iloc[:top], stats.iloc[top:]
        other = pd.DataFrame(
            {"applicants": [tail["applicants"].sum()],
             "default_rate": [np.average(tail["default_rate"], weights=tail["applicants"])]},
            index=["Other"],
        )
        stats = pd.concat([head, other])
    total = stats["applicants"].sum()
    return {
        "key": key,
        "title": title,
        "description": description,
        "ordered": ordered,
        "rows": [
            {"segment": str(name), "applicants": int(r.applicants), "share": round(r.applicants / total, 5),
             "default_rate": round(float(r.default_rate), 5)}
            for name, r in stats.iterrows()
        ],
    }


def build_insights(base: pd.DataFrame) -> dict:
    df = add_derived_features(base)
    y = df[TARGET_COLUMN].astype(int)
    years_employed = -df["DAYS_EMPLOYED"] / 365.25
    ext_mean = df["EXT_SOURCE_MEAN"]
    ext_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]
    ext_quintile = pd.qcut(ext_mean, 5, labels=ext_labels).astype(object).where(ext_mean.notna(), "Not available")
    late = df["INSTAL_LATE_SHARE"]
    late_group = _bins(late, [0, 1e-9, 0.05, 0.15, INF], ["Never late", "Under 5%", "5-15%", "15% or more"], "No history")
    refused = df["PREV_REFUSED_COUNT"]
    refused_group = _bins(refused, [0, 1, 2, 4, INF], ["None", "1", "2-3", "4 or more"], "New client")

    categorical = lambda col: df[col].astype(object).where(df[col].notna(), "Not stated")  # noqa: E731
    segments = [
        _segment("income", "Annual income", _bins(df["AMT_INCOME_TOTAL"], [0, 100e3, 150e3, 200e3, 300e3, INF],
                 ["Under 100k", "100k-150k", "150k-200k", "200k-300k", "300k or more"]), y, True,
                 "Total annual income of the applicant."),
        _segment("age", "Age", age_band(df["DAYS_BIRTH"]).astype(object), y, True, "Applicant age at application.",
                 order=["Under 30", "30-39", "40-49", "50-59", "60+"]),
        _segment("employment", "Years in current job", _bins(years_employed, [0, 1, 3, 5, 10, INF],
                 ["Under 1", "1-3", "3-5", "5-10", "10 or more"], "Not employed"), y, True,
                 "Pensioners and unemployed applicants are shown as 'Not employed'.",
                 order=["Not employed", "Under 1", "1-3", "3-5", "5-10", "10 or more"]),
        _segment("ext_score", "External score (average)", ext_quintile, y, True,
                 "Average of the three external credit scores, split into quintiles.",
                 order=["Not available", *ext_labels]),
        _segment("credit_income", "Credit-to-income ratio", _bins(df["CREDIT_INCOME_RATIO"], [0, 2, 3, 4, 6, INF],
                 ["Under 2x", "2-3x", "3-4x", "4-6x", "6x or more"]), y, True, "Requested credit divided by annual income."),
        _segment("education", "Education", categorical("NAME_EDUCATION_TYPE"), y, False, "Highest education reached."),
        _segment("income_type", "Income type", categorical("NAME_INCOME_TYPE"), y, False, "Main source of income.", top=6),
        _segment("family", "Family status", categorical("NAME_FAMILY_STATUS"), y, False, "Marital / family status.", top=5),
        _segment("housing", "Housing situation", categorical("NAME_HOUSING_TYPE"), y, False, "Current housing arrangement."),
        _segment("occupation", "Occupation", categorical("OCCUPATION_TYPE"), y, False, "Top occupations; the rest are grouped.", top=12),
        _segment("contract", "Contract type", categorical("NAME_CONTRACT_TYPE"), y, False, "Cash loan or revolving credit."),
        _segment("region", "Region rating", df["REGION_RATING_CLIENT"].map(lambda v: f"Rating {v:.0f}" if pd.notna(v) else None),
                 y, True, "Home Credit's rating of the applicant's region (1 is best).",
                 order=["Rating 1", "Rating 2", "Rating 3"]),
        _segment("bureau", "Credit bureau loans", _bins(df["BUREAU_LOAN_COUNT"], [1, 3, 6, 11, INF],
                 ["1-2", "3-5", "6-10", "11 or more"], "No history"), y, True,
                 "Loans with other lenders reported to the credit bureau.",
                 order=["No history", "1-2", "3-5", "6-10", "11 or more"]),
        _segment("late_payments", "Late installment share", late_group, y, True,
                 "Share of previous Home Credit installments paid after the due date.",
                 order=["No history", "Never late", "Under 5%", "5-15%", "15% or more"]),
        _segment("refusals", "Previous refusals", refused_group, y, True,
                 "Previous Home Credit applications that were refused.",
                 order=["New client", "None", "1", "2-3", "4 or more"]),
    ]
    kpis = {
        "applicants": int(len(df)),
        "defaults": int(y.sum()),
        "default_rate": round(float(y.mean()), 5),
        "median_income": float(df["AMT_INCOME_TOTAL"].median()),
        "median_credit": float(df["AMT_CREDIT"].median()),
        "median_annuity": float(df["AMT_ANNUITY"].median()),
        "median_age": round(float((-df["DAYS_BIRTH"] / 365.25).median()), 1),
        "share_with_bureau_history": round(float(df["BUREAU_LOAN_COUNT"].notna().mean()), 4),
        "share_repeat_clients": round(float(df["PREV_APP_COUNT"].notna().mean()), 4),
    }
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kpis": kpis, "segments": segments}


def save_insights(insights: dict, model_dir: Path | None = None) -> Path:
    path = Path(model_dir or config.model_dir()) / config.INSIGHTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    log.info("Wrote portfolio insights (%d segments) to %s", len(insights["segments"]), path)
    return path
