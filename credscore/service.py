"""Scoring service: applicant data in, decision-ready results out.

Used by the FastAPI backend, and directly by tests and notebooks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import config
from .features import DERIVED_FEATURES
from .labels import feature_label, format_value
from .model import CreditModel
from .profile import CATEGORICAL_FIELDS, ApplicantProfile, profile_defaults, profile_to_base_features
from .scoring import DecisionPolicy, pd_to_score, risk_band, scorecard_description

KEY_METRICS = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "PAYMENT_RATE", "EXT_SOURCE_MEAN"]


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else round(float(value), 6)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


class ScoringService:
    def __init__(self, model: CreditModel, model_dir: Path | None = None):
        self.model = model
        self.spec = model.spec
        self.model_dir = Path(model_dir) if model_dir else None
        stored = model.metadata["policy"]
        self.policy = DecisionPolicy(
            approve_below=config.env_float("CREDSCORE_APPROVE_PD") or stored["approve_below"],
            decline_at=config.env_float("CREDSCORE_DECLINE_PD") or stored["decline_at"],
        )
        self._feature_index = {f: i for i, f in enumerate(self.spec.feature_names)}

    @classmethod
    def load(cls, model_dir: Path | None = None) -> "ScoringService":
        model_dir = Path(model_dir or config.model_dir())
        return cls(CreditModel.load(model_dir), model_dir)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def profile_errors(self, profile: ApplicantProfile) -> list[str]:
        """Categorical values the model has never seen (the vocabulary is data-driven)."""
        errors = []
        for field, feature in CATEGORICAL_FIELDS.items():
            value = getattr(profile, field)
            allowed = self.spec.categories.get(feature)
            if value is not None and allowed is not None and value not in allowed:
                errors.append(f"{field}: '{value}' is not one of {allowed}")
        return errors

    def unknown_features(self, features: dict) -> list[str]:
        base = set(self.spec.base_features)
        return sorted(f for f in features if f not in base)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score_profiles(self, profiles: list[ApplicantProfile], top_k: int = 8, n_reasons: int = 4) -> list[dict]:
        return self.score_records([profile_to_base_features(p) for p in profiles], top_k, n_reasons)

    def score_records(self, records: list[dict], top_k: int = 8, n_reasons: int = 4) -> list[dict]:
        """Score base-feature records; see `FeatureSpec.frame_from_records` for missing keys."""
        if not records:
            return []
        records = [{k: v for k, v in r.items() if k in self._feature_index} for r in records]
        X = self.spec.frame_from_records(records)
        pds = self.model.predict_pd(X)
        explain = top_k > 0 or n_reasons > 0
        contributions = self.model.contributions(X) if explain else None
        values = X.astype(object).to_numpy()
        results = []
        for i, record in enumerate(records):
            result = self._decision(float(pds[i]))
            result["key_metrics"] = {
                f: self._feature_view(f, values[i]) for f in KEY_METRICS if f in self._feature_index
            }
            if contributions is not None:
                result |= self._explain(values[i], contributions[i], set(record), top_k, n_reasons)
            results.append(result)
        return results

    def _decision(self, pd_value: float) -> dict:
        score = pd_to_score(pd_value)
        return {
            "probability_of_default": round(pd_value, 6),
            "credit_score": score,
            "risk_band": asdict(risk_band(score)),
            "decision": self.policy.decide(pd_value),
            "decision_reason": self.policy.explain(pd_value),
            "model_version": self.model.version,
        }

    def _feature_view(self, feature: str, row_values) -> dict:
        value = _json_value(row_values[self._feature_index[feature]])
        return {"label": feature_label(feature), "value": value, "display_value": format_value(feature, value)}

    def _explain(self, row_values, contributions, provided: set, top_k: int, n_reasons: int) -> dict:
        feature_contribs = contributions[:-1]
        order = np.argsort(-np.abs(feature_contribs))
        explanations = []
        for j in order[: max(top_k, n_reasons * 3)]:
            feature = self.spec.feature_names[j]
            contribution = float(feature_contribs[j])
            if contribution == 0:
                continue
            if feature in DERIVED_FEATURES:
                source = "derived"
            else:
                source = "applicant" if feature in provided else "default"
            explanations.append(
                {"feature": feature, **self._feature_view(feature, row_values),
                 "contribution": round(contribution, 5),
                 "effect": "increases_risk" if contribution > 0 else "decreases_risk",
                 "source": source}
            )
        reasons = [
            f"{e['label']}: {e['display_value']}" for e in explanations if e["effect"] == "increases_risk"
        ][:n_reasons]
        return {
            "explanations": explanations[:top_k],
            "reasons": reasons,
            "base_log_odds": round(float(contributions[-1]), 5),
        }

    # ------------------------------------------------------------------ #
    # Metadata for the API / UI
    # ------------------------------------------------------------------ #
    def model_info(self) -> dict:
        meta = self.model.metadata
        return {
            "version": self.model.version,
            "created_at": meta.get("created_at"),
            "algorithm": meta.get("algorithm"),
            "n_features": len(self.spec.feature_names),
            "n_base_features": len(self.spec.base_features),
            "training": meta.get("training", {}),
            "metrics": meta.get("metrics", {}),
            "policy": asdict(self.policy),
            "policy_targets": meta.get("policy", {}).get("targets"),
            "scorecard": scorecard_description(),
            "excluded_features": meta.get("excluded_features", []),
        }

    def schema(self) -> dict:
        return {
            "profile": ApplicantProfile.model_json_schema(),
            "options": {
                field: self.spec.categories.get(feature, []) for field, feature in CATEGORICAL_FIELDS.items()
            },
            "defaults": profile_defaults(self.spec.defaults),
            "base_features": [
                {
                    "name": f,
                    "label": feature_label(f),
                    "categorical": f in self.spec.categories,
                    "categories": self.spec.categories.get(f),
                    "default": self.spec.defaults.get(f),
                    "missing_rate": self.spec.missing_rates.get(f),
                }
                for f in self.spec.base_features
            ],
        }

    def _artifact(self, name: str) -> dict | None:
        if self.model_dir is None or not (self.model_dir / name).exists():
            return None
        return json.loads((self.model_dir / name).read_text(encoding="utf-8"))

    def performance_report(self) -> dict | None:
        return self._artifact(config.REPORT_FILE)

    def insights(self) -> dict | None:
        return self._artifact(config.INSIGHTS_FILE)
