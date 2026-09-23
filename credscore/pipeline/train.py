"""Train, evaluate and package the credit risk model.

Design choices, versus a naive SMOTE + label-encoding setup:

* The data is split into train / validation / test **before** anything is
  learned from it, and every reported number comes from the untouched test set.
* No resampling. Class imbalance (8% defaults) is left as is, so predicted
  probabilities stay calibrated to real default rates, which the scorecard and
  decision policy depend on. Oversampling mainly buys miscalibration.
* Missing values stay missing (XGBoost learns where to send them) and
  categoricals use XGBoost's native categorical splits, so no imputer or
  encoder has to be replicated at serving time.
* Early stopping on the validation set picks the number of trees; the
  validation set also sets the decision-policy cut-offs.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .. import config, evaluation
from ..features import GENDER_COLUMN, TARGET_COLUMN, FeatureSpec
from ..labels import feature_label
from ..model import CreditModel
from ..scoring import DecisionPolicy

log = logging.getLogger(__name__)

EXCLUDED_FEATURES = [
    {"feature": "CODE_GENDER", "reason": "Protected attribute; used only for fairness monitoring."},
    {"feature": "WEEKDAY_APPR_PROCESS_START, HOUR_APPR_PROCESS_START", "reason": "Application-process artefacts, not applicant traits."},
    {"feature": "FLAG_DOCUMENT_* (except 3), FLAG_MOBIL", "reason": "Near-constant."},
    {"feature": "Building *_MODE / *_MEDI", "reason": "Duplicates of the *_AVG building statistics."},
]


@dataclass
class TrainConfig:
    device: str = "auto"
    learning_rate: float = 0.03
    max_depth: int = 6
    min_child_weight: float = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.5
    reg_lambda: float = 5.0
    num_boost_round: int = 5000
    early_stopping_rounds: int = 200
    valid_size: float = 0.15
    test_size: float = 0.15
    seed: int = config.RANDOM_STATE
    log_every: int = 250

    def xgb_params(self, device: str) -> dict:
        return {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],  # early stopping uses the last one
            "tree_method": "hist",
            "device": device,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "seed": self.seed,
            "verbosity": 1,
        }


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if not xgb.build_info().get("USE_CUDA"):
        return "cpu"
    try:
        X = np.random.default_rng(0).random((64, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a CPU fallback warning means no usable GPU
            xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0}, xgb.DMatrix(X, label=X[:, 0] > 0.5), 1)
        return "cuda"
    except Exception:
        return "cpu"


def split_indices(y: np.ndarray, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    holdout = cfg.valid_size + cfg.test_size
    train_idx, rest_idx = train_test_split(idx, test_size=holdout, stratify=y, random_state=cfg.seed)
    valid_idx, test_idx = train_test_split(
        rest_idx, test_size=cfg.test_size / holdout, stratify=y[rest_idx], random_state=cfg.seed
    )
    return train_idx, valid_idx, test_idx


def global_importance(model: CreditModel, X: pd.DataFrame, gain: dict, top: int = 30) -> list[dict]:
    """Mean |SHAP| per feature (log-odds), plus total gain share for reference."""
    contributions = model.contributions(X)[:, :-1]
    mean_abs = np.abs(contributions).mean(axis=0)
    total_gain = sum(gain.values()) or 1.0
    order = np.argsort(-mean_abs)[:top]
    return [
        {
            "feature": model.spec.feature_names[j],
            "label": feature_label(model.spec.feature_names[j]),
            "mean_abs_shap": round(float(mean_abs[j]), 5),
            "gain_share": round(gain.get(model.spec.feature_names[j], 0.0) / total_gain, 5),
        }
        for j in order
    ]


def train(base: pd.DataFrame, cfg: TrainConfig | None = None) -> tuple[CreditModel, dict]:
    cfg = cfg or TrainConfig()
    started = time.perf_counter()
    y = base[TARGET_COLUMN].astype(int).to_numpy()
    train_idx, valid_idx, test_idx = split_indices(y, cfg)
    log.info("Split: %s train / %s valid / %s test", f"{len(train_idx):,}", f"{len(valid_idx):,}", f"{len(test_idx):,}")

    spec = FeatureSpec.fit(base.iloc[train_idx])
    X_train, X_valid, X_test = (spec.transform(base.iloc[i]) for i in (train_idx, valid_idx, test_idx))
    y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]
    log.info("Features: %d base + %d derived", len(spec.base_features), len(spec.feature_names) - len(spec.base_features))

    device = resolve_device(cfg.device)
    params = cfg.xgb_params(device)
    log.info("Training XGBoost on %s (lr=%s, max_depth=%s)...", device.upper(), cfg.learning_rate, cfg.max_depth)
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose_eval=cfg.log_every or False,
    )
    best_iteration = booster.best_iteration
    booster = booster[: best_iteration + 1]
    booster.set_param({"device": "cpu"})
    train_seconds = time.perf_counter() - started
    log.info("Best iteration %d (%.0fs)", best_iteration, train_seconds)

    model = CreditModel(booster, spec, metadata={})
    p_train, p_valid, p_test = (model.predict_pd(X) for X in (X_train, X_valid, X_test))

    policy = DecisionPolicy(
        approve_below=round(float(np.quantile(p_valid, config.APPROVE_RATE)), 5),
        decline_at=round(float(np.quantile(p_valid, 1 - config.DECLINE_RATE)), 5),
    )

    # Baseline: the average of the external scores alone (a common strong benchmark).
    baseline_train = X_train["EXT_SOURCE_MEAN"].astype(float)
    baseline_test = -X_test["EXT_SOURCE_MEAN"].astype(float).fillna(baseline_train.median()).to_numpy()

    metrics = {
        "test": evaluation.classification_metrics(y_test, p_test),
        "valid": evaluation.classification_metrics(y_valid, p_valid),
        "train": evaluation.classification_metrics(y_train, p_train),
        "baseline_test": evaluation.ranking_metrics(y_test, baseline_test),
    }
    log.info(
        "Test ROC AUC %.4f (baseline %.4f) | Gini %.4f | KS %.4f | Brier %.5f",
        metrics["test"]["roc_auc"], metrics["baseline_test"]["roc_auc"], metrics["test"]["gini"],
        metrics["test"]["ks"], metrics["test"]["brier"],
    )

    version = datetime.now().strftime("%Y%m%d-%H%M%S")
    model.metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": f"XGBoost {xgb.__version__} gradient-boosted trees (native categorical splits)",
        "training": {
            "n_rows": int(len(base)),
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
            "n_test": int(len(test_idx)),
            "base_rate": round(float(y.mean()), 5),
            "best_iteration": int(best_iteration),
            "device": device,
            "train_seconds": round(train_seconds, 1),
            "config": asdict(cfg),
        },
        "metrics": metrics,
        "policy": asdict(policy) | {"targets": {"approve_rate": config.APPROVE_RATE, "decline_rate": config.DECLINE_RATE}},
        "excluded_features": EXCLUDED_FEATURES,
    }

    sample = X_test.sample(n=min(10_000, len(X_test)), random_state=cfg.seed)
    test_rows = base.iloc[test_idx]
    report = {
        "version": version,
        "metrics": metrics,
        "roc": {"model": evaluation.roc_points(y_test, p_test), "baseline": evaluation.roc_points(y_test, baseline_test)},
        "pr": evaluation.pr_points(y_test, p_test),
        "calibration": evaluation.calibration_table(y_test, p_test),
        "gains": evaluation.gains_table(y_test, p_test),
        "bands": evaluation.band_table(y_test, p_test),
        "policy": {"cutoffs": asdict(policy), "outcomes": evaluation.policy_table(y_test, p_test, policy)},
        "score_histogram": evaluation.score_histogram(y_test, p_test),
        "importance": global_importance(model, sample, booster.get_score(importance_type="total_gain")),
        "fairness": {
            "gender": evaluation.group_report(y_test, p_test, test_rows[GENDER_COLUMN], policy)
            if GENDER_COLUMN in test_rows else [],
            "age": evaluation.group_report(y_test, p_test, evaluation.age_band(test_rows["DAYS_BIRTH"]), policy),
        },
    }
    return model, report


def save(model: CreditModel, report: dict, model_dir: Path | None = None) -> Path:
    import json

    model_dir = Path(model_dir or config.model_dir())
    model.save(model_dir)
    (model_dir / config.REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Saved model bundle %s to %s", model.version, model_dir)
    return model_dir
