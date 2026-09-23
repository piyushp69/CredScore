"""Paths and tunable settings.

Directories can be overridden with environment variables so the same code runs
locally, in tests (temporary model directories) and in containers.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

FEATURES_FILE = "features.parquet"
MODEL_FILE = "model.ubj"
METADATA_FILE = "metadata.json"
REPORT_FILE = "report.json"
INSIGHTS_FILE = "insights.json"

RAW_TABLES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "previous": "previous_application.csv",
    "installments": "installments_payments.csv",
}
KAGGLE_COMPETITION = "home-credit-default-risk"

RANDOM_STATE = 42

# Decision policy targets, applied to the validation-set PD distribution at
# training time: the safest APPROVE_RATE of applicants are approved, the riskiest
# DECLINE_RATE are declined, and everyone in between goes to manual review.
APPROVE_RATE = 0.70
DECLINE_RATE = 0.10


def data_dir() -> Path:
    return Path(os.environ.get("CREDSCORE_DATA_DIR", ROOT_DIR / "dataset")).resolve()


def model_dir() -> Path:
    return Path(os.environ.get("CREDSCORE_MODEL_DIR", ROOT_DIR / "models")).resolve()


def env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else None
