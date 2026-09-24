"""Scoring service: applicant data in, decision-ready results out.

Used by the FastAPI backend, the Streamlit dashboard, tests and notebooks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from functools import cached_property
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from . import config
from .features import DERIVED_FEATURES
from .labels import feature_label, format_value
from .model import CreditModel
from .profile import CATEGORICAL_FIELDS, ApplicantProfile, profile_defaults, profile_to_base_features
from .scoring import RISK_BANDS, DecisionPolicy, pd_to_score, risk_band, scorecard_description

KEY_METRICS = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "PAYMENT_RATE", "EXT_SOURCE_MEAN"]
_BAND_DICTS = {band.code: asdict(band) for band in RISK_BANDS}


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else round(float(value), 6)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def _clean_row(row: dict) -> dict:
    """CSV-friendly: NaN and empty strings mean 'not provided'."""
    return {
        k: None if (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip()) else v
        for k, v in row.items()
    }


class ScoringService:
    def __init__(self, model: CreditModel, model_dir: Path | None = None):
        self.model = model
        self.spec = model.spec
        self.model_dir = Path(model_dir) if model_dir else None
        stored = model.metadata["policy"]
        approve_below = config.env_float("CREDSCORE_APPROVE_PD")
        decline_at = config.env_float("CREDSCORE_DECLINE_PD")
        self.policy = DecisionPolicy(
            approve_below=stored["approve_below"] if approve_below is None else approve_below,
            decline_at=stored["decline_at"] if decline_at is None else decline_at,
        )
        self._feature_index = {f: i for i, f in enumerate(self.spec.feature_names)}
        self._base_features = set(self.spec.base_features)
        self._labels = {f: feature_label(f) for f in self.spec.feature_names}
        self._allowed = {feature: set(values) for feature, values in self.spec.categories.items()}

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
            allowed = self._allowed.get(feature)
            if value is not None and allowed is not None and value not in allowed:
                errors.append(f"{field}: '{value}' is not one of {self.spec.categories[feature]}")
        return errors

    def unknown_features(self, features: dict) -> list[str]:
        return sorted(f for f in features if f not in self._base_features)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score_profiles(self, profiles: list[ApplicantProfile], top_k: int = 8, n_reasons: int = 4) -> list[dict]:
        return self.score_records([profile_to_base_features(p) for p in profiles], top_k, n_reasons)

    def score_records(self, records: list[dict], top_k: int = 8, n_reasons: int = 4) -> list[dict]:
        """Score base-feature records; see `FeatureSpec.frame_from_records` for missing keys."""
        if not records:
            return []
        records = [{k: v for k, v in r.items() if k in self._base_features} for r in records]
        X = self.spec.frame_from_records(records)
        explain = top_k > 0 or n_reasons > 0
        if explain:
            pds, contributions = self.model.predict_with_contributions(X)
            # Only the first `n_candidates` features by |SHAP| can become explanations or reasons.
            n_candidates = min(max(top_k, n_reasons * 3), len(self.spec.feature_names))
            candidates = np.argsort(-np.abs(contributions[:, :-1]), axis=1, kind="stable")[:, :n_candidates]
        else:
            pds = self.model.predict_pd(X)
        scores = pd_to_score(pds)
        # Column arrays are views of X; reading single cells from them avoids
        # copying the whole matrix into Python objects.
        columns = [X[f].to_numpy() for f in self.spec.feature_names]
        key_metrics = [f for f in KEY_METRICS if f in self._feature_index]
        results = []
        for i, record in enumerate(records):
            result = self._decision(float(pds[i]), int(scores[i]))
            result["key_metrics"] = {f: self._feature_view(f, columns, i) for f in key_metrics}
            if explain:
                result |= self._explain(columns, i, contributions[i], candidates[i], record.keys(), top_k, n_reasons)
            results.append(result)
        return results

    def score_batch(self, rows: list[dict], n_reasons: int = 3) -> dict:
        """Score raw applicant rows (e.g. parsed CSV). Invalid rows are reported, not fatal.

        Each row may carry an `applicant_id`; NaN and blank values mean "not provided".
        Returns plain dicts: `{"summary": {...}, "results": [{index, applicant_id, result, error}]}`.
        """
        items: list[dict] = []
        valid: list[tuple[int, ApplicantProfile]] = []
        for index, raw in enumerate(rows):
            row = _clean_row(raw)
            applicant_id = row.pop("applicant_id", None)
            item = {"index": index, "applicant_id": None if applicant_id is None else str(applicant_id),
                    "result": None, "error": None}
            try:
                profile = ApplicantProfile.model_validate(row)
                errors = self.profile_errors(profile)
            except ValidationError as exc:
                errors = [f"{'.'.join(map(str, e['loc'])) or 'row'}: {e['msg']}" for e in exc.errors()]
            if errors:
                item["error"] = "; ".join(errors)
            else:
                valid.append((index, profile))
            items.append(item)

        results = self.score_profiles([p for _, p in valid], top_k=0, n_reasons=n_reasons)
        for (index, _), result in zip(valid, results):
            items[index]["result"] = result

        decisions = {"APPROVE": 0, "REVIEW": 0, "DECLINE": 0}
        for result in results:
            decisions[result["decision"]] += 1
        n = len(results)
        summary = {
            "submitted": len(items),
            "scored": n,
            "failed": len(items) - n,
            "decisions": decisions,
            "mean_probability_of_default": round(sum(r["probability_of_default"] for r in results) / n, 6) if n else None,
            "mean_credit_score": round(sum(r["credit_score"] for r in results) / n, 1) if n else None,
        }
        return {"summary": summary, "results": items}

    def _decision(self, pd_value: float, score: int) -> dict:
        return {
            "probability_of_default": round(pd_value, 6),
            "credit_score": score,
            "risk_band": dict(_BAND_DICTS[risk_band(score).code]),
            "decision": self.policy.decide(pd_value),
            "decision_reason": self.policy.explain(pd_value),
            "model_version": self.model.version,
        }

    def _feature_view(self, feature: str, columns: list, row: int) -> dict:
        value = _json_value(columns[self._feature_index[feature]][row])
        return {"label": self._labels[feature], "value": value, "display_value": format_value(feature, value)}

    def _explain(self, columns: list, row: int, contributions, candidates, provided, top_k: int, n_reasons: int) -> dict:
        """Top-|SHAP| explanations and the leading risk-raising reasons.

        Views are only built for features that are actually returned; batch
        scoring asks for reasons alone, so most candidates are skipped cheaply.
        """
        explanations, reasons = [], []
        for j in candidates:
            contribution = float(contributions[j])
            if contribution == 0:
                continue
            as_explanation = len(explanations) < top_k
            as_reason = contribution > 0 and len(reasons) < n_reasons
            if not (as_explanation or as_reason):
                continue
            feature = self.spec.feature_names[j]
            view = self._feature_view(feature, columns, row)
            if as_reason:
                reasons.append(f"{view['label']}: {view['display_value']}")
            if as_explanation:
                if feature in DERIVED_FEATURES:
                    source = "derived"
                else:
                    source = "applicant" if feature in provided else "default"
                explanations.append(
                    {"feature": feature, **view,
                     "contribution": round(contribution, 5),
                     "effect": "increases_risk" if contribution > 0 else "decreases_risk",
                     "source": source}
                )
        return {
            "explanations": explanations,
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
        return self._schema

    @cached_property
    def _schema(self) -> dict:
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

    # Artifacts belong to the loaded model bundle, so they are read once, like the model.
    @cached_property
    def _report(self) -> dict | None:
        return self._artifact(config.REPORT_FILE)

    @cached_property
    def _insights(self) -> dict | None:
        return self._artifact(config.INSIGHTS_FILE)

    def performance_report(self) -> dict | None:
        return self._report

    def insights(self) -> dict | None:
        return self._insights
