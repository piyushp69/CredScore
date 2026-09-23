"""The trained model bundle: XGBoost booster + feature spec + metadata.

The booster is stored in XGBoost's native UBJSON format rather than pickled,
so bundles stay loadable across library versions and cannot execute code on
load. Per-feature explanations come from XGBoost's built-in TreeSHAP
(`pred_contribs=True`), so no separate SHAP explainer has to be shipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import METADATA_FILE, MODEL_FILE
from .features import FeatureSpec


class ModelNotFoundError(FileNotFoundError):
    pass


class CreditModel:
    def __init__(self, booster: xgb.Booster, spec: FeatureSpec, metadata: dict):
        self.booster = booster
        self.spec = spec
        self.metadata = metadata

    @property
    def version(self) -> str:
        return self.metadata.get("version", "unknown")

    def _dmatrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(X, enable_categorical=True)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._dmatrix(X))

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values in log-odds space, shape (n_rows, n_features + 1); last column is the bias."""
        return self.booster.predict(self._dmatrix(X), pred_contribs=True)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / MODEL_FILE))
        metadata = self.metadata | {"feature_spec": self.spec.to_dict()}
        (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "CreditModel":
        directory = Path(directory)
        model_path, metadata_path = directory / MODEL_FILE, directory / METADATA_FILE
        if not model_path.exists() or not metadata_path.exists():
            raise ModelNotFoundError(
                f"No trained model in {directory}. Run `python -m credscore.pipeline all` first."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        spec = FeatureSpec.from_dict(metadata.pop("feature_spec"))
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        booster.set_param({"device": "cpu"})
        if booster.feature_names != spec.feature_names:
            raise ValueError("Model features do not match the stored feature spec; retrain the model.")
        return cls(booster, spec, metadata)
