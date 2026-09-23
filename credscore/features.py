"""Feature engineering shared by the training pipeline and the scoring API.

Features live in two layers:

* **Base features**: raw application columns plus per-applicant aggregates of
  the bureau, previous-application and installment tables. Callers supply
  these, either directly or through an `ApplicantProfile`.
* **Derived features**: ratios and summaries computed from base features by
  `add_derived_features`. They are always recomputed and never accepted as
  input, so training and serving cannot drift apart.

`FeatureSpec` captures everything learned from the training split (feature
order, category vocabularies, serving defaults) and turns base features into
the exact matrix the model was trained on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

DAYS_EMPLOYED_ANOMALY = 365243  # Home Credit's placeholder for "not employed"
EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# Columns carried through the feature table that are never model inputs.
ID_COLUMN, TARGET_COLUMN, GENDER_COLUMN = "SK_ID_CURR", "TARGET", "CODE_GENDER"
NON_FEATURE_COLUMNS = {ID_COLUMN, TARGET_COLUMN, GENDER_COLUMN}

EXCLUDED_APPLICATION_COLUMNS = NON_FEATURE_COLUMNS | {
    # Properties of the application process rather than the applicant.
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    # Constant in practice.
    "FLAG_MOBIL",
}
KEPT_DOCUMENT_FLAGS = {"FLAG_DOCUMENT_3"}  # the other document flags are near-constant

CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]
YES_NO_COLUMNS = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]

# Raw columns each side table needs; used to read the CSVs cheaply.
BUREAU_COLUMNS = [
    "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
    "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
]
PREVIOUS_COLUMNS = [
    "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT",
    "AMT_ANNUITY", "DAYS_DECISION", "CNT_PAYMENT",
]
INSTALLMENT_COLUMNS = [
    "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
]

BUREAU_AGGREGATIONS = {
    "BUREAU_LOAN_COUNT": ("SK_ID_BUREAU", "count"),
    "BUREAU_ACTIVE_COUNT": ("IS_ACTIVE", "sum"),
    "BUREAU_DAYS_CREDIT_MEAN": ("DAYS_CREDIT", "mean"),
    "BUREAU_DAYS_CREDIT_MAX": ("DAYS_CREDIT", "max"),
    "BUREAU_DAYS_CREDIT_ENDDATE_MEAN": ("DAYS_CREDIT_ENDDATE", "mean"),
    "BUREAU_CREDIT_DAY_OVERDUE_MAX": ("CREDIT_DAY_OVERDUE", "max"),
    "BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX": ("AMT_CREDIT_MAX_OVERDUE", "max"),
    "BUREAU_AMT_CREDIT_SUM": ("AMT_CREDIT_SUM", "sum"),
    "BUREAU_AMT_CREDIT_SUM_DEBT": ("AMT_CREDIT_SUM_DEBT", "sum"),
    "BUREAU_AMT_CREDIT_SUM_OVERDUE": ("AMT_CREDIT_SUM_OVERDUE", "sum"),
    "BUREAU_CNT_CREDIT_PROLONG": ("CNT_CREDIT_PROLONG", "sum"),
}
PREVIOUS_AGGREGATIONS = {
    "PREV_APP_COUNT": ("SK_ID_PREV", "count"),
    "PREV_APPROVED_COUNT": ("IS_APPROVED", "sum"),
    "PREV_REFUSED_COUNT": ("IS_REFUSED", "sum"),
    "PREV_AMT_APPLICATION_MEAN": ("AMT_APPLICATION", "mean"),
    "PREV_AMT_CREDIT_MEAN": ("AMT_CREDIT", "mean"),
    "PREV_AMT_ANNUITY_MEAN": ("AMT_ANNUITY", "mean"),
    "PREV_APP_CREDIT_RATIO_MEAN": ("APP_CREDIT_RATIO", "mean"),
    "PREV_DAYS_DECISION_MEAN": ("DAYS_DECISION", "mean"),
    "PREV_DAYS_DECISION_MAX": ("DAYS_DECISION", "max"),
    "PREV_CNT_PAYMENT_MEAN": ("CNT_PAYMENT", "mean"),
}
INSTALLMENT_AGGREGATIONS = {
    "INSTAL_COUNT": ("DAYS_INSTALMENT", "count"),
    "INSTAL_DPD_MEAN": ("DPD", "mean"),
    "INSTAL_DPD_MAX": ("DPD", "max"),
    "INSTAL_DBD_MEAN": ("DBD", "mean"),
    "INSTAL_LATE_SHARE": ("IS_LATE", "mean"),
    "INSTAL_PAYMENT_RATIO_MEAN": ("PAYMENT_RATIO", "mean"),
    "INSTAL_PAYMENT_SHORTFALL_MEAN": ("PAYMENT_SHORTFALL", "mean"),
    "INSTAL_AMT_PAYMENT_SUM": ("AMT_PAYMENT", "sum"),
}

# Aggregates are all-missing for applicants with no rows in the side table.
FEATURE_GROUPS = {
    "bureau": list(BUREAU_AGGREGATIONS),
    "previous": list(PREVIOUS_AGGREGATIONS),
    "installments": list(INSTALLMENT_AGGREGATIONS),
}

DERIVED_FEATURES = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "PAYMENT_RATE",
    "CREDIT_GOODS_RATIO",
    "INCOME_PER_PERSON",
    "EMPLOYED_AGE_RATIO",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_MAX",
    "EXT_SOURCE_STD",
    "BUREAU_ACTIVE_RATIO",
    "BUREAU_DEBT_CREDIT_RATIO",
    "BUREAU_DEBT_INCOME_RATIO",
    "PREV_REFUSED_RATIO",
    "PREV_APPROVED_RATIO",
]

# A feature missing in at least this share of training rows defaults to
# "missing" at serving time; otherwise it defaults to the median / mode.
MISSING_DEFAULT_THRESHOLD = 0.4


# --------------------------------------------------------------------------- #
# Raw tables -> base features
# --------------------------------------------------------------------------- #
def is_model_application_column(column: str) -> bool:
    if column in EXCLUDED_APPLICATION_COLUMNS:
        return False
    if column.startswith("FLAG_DOCUMENT_") and column not in KEPT_DOCUMENT_FLAGS:
        return False
    # Building information is repeated as _AVG, _MODE and _MEDI; keep _AVG only.
    if column.endswith("_MEDI"):
        return False
    if column.endswith("_MODE") and column not in CATEGORICAL_FEATURES and column != "TOTALAREA_MODE":
        return False
    return True


def clean_application(app: pd.DataFrame) -> pd.DataFrame:
    app = app.copy()
    for column in YES_NO_COLUMNS:
        if column in app:
            app[column] = app[column].map({"Y": 1.0, "N": 0.0})
    if "DAYS_EMPLOYED" in app:
        app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].where(app["DAYS_EMPLOYED"] != DAYS_EMPLOYED_ANOMALY)
    if GENDER_COLUMN in app:
        app[GENDER_COLUMN] = app[GENDER_COLUMN].where(app[GENDER_COLUMN].isin(["F", "M"]))
    return app


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator != 0)


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    bureau = bureau.assign(IS_ACTIVE=(bureau["CREDIT_ACTIVE"] == "Active").astype(float))
    return bureau.groupby(ID_COLUMN).agg(**BUREAU_AGGREGATIONS)


def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    status = prev["NAME_CONTRACT_STATUS"]
    prev = prev.assign(
        IS_APPROVED=(status == "Approved").astype(float),
        IS_REFUSED=(status == "Refused").astype(float),
        APP_CREDIT_RATIO=_ratio(prev["AMT_APPLICATION"], prev["AMT_CREDIT"]),
    )
    return prev.groupby(ID_COLUMN).agg(**PREVIOUS_AGGREGATIONS)


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    days_late = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst = inst.assign(
        DPD=days_late.clip(lower=0),
        DBD=(-days_late).clip(lower=0),
        IS_LATE=(days_late > 0).astype(float).where(days_late.notna()),
        PAYMENT_RATIO=_ratio(inst["AMT_PAYMENT"], inst["AMT_INSTALMENT"]),
        PAYMENT_SHORTFALL=inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"],
    )
    return inst.groupby(ID_COLUMN).agg(**INSTALLMENT_AGGREGATIONS)


def build_base_table(
    application: pd.DataFrame,
    bureau: pd.DataFrame,
    previous: pd.DataFrame,
    installments: pd.DataFrame,
) -> pd.DataFrame:
    """One row per applicant: id, target, gender (for fairness reports) and base features."""
    app = clean_application(application)
    keep = [c for c in app.columns if c in NON_FEATURE_COLUMNS or is_model_application_column(c)]
    base = app[keep].set_index(ID_COLUMN)
    for aggregates in (
        aggregate_bureau(bureau),
        aggregate_previous(previous),
        aggregate_installments(installments),
    ):
        base = base.join(aggregates, how="left")

    for column in base.columns:
        if column in CATEGORICAL_FEATURES or column == GENDER_COLUMN:
            base[column] = base[column].astype("category")
        elif column != TARGET_COLUMN:
            base[column] = base[column].astype("float32")
    return base.reset_index()


# --------------------------------------------------------------------------- #
# Base features -> derived features
# --------------------------------------------------------------------------- #
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    def col(name: str) -> pd.Series:
        if name in df:
            return df[name].astype(float)
        return pd.Series(np.nan, index=df.index)

    income = col("AMT_INCOME_TOTAL")
    ext = pd.concat([col(c) for c in EXT_SOURCES], axis=1)
    derived = {
        "CREDIT_INCOME_RATIO": _ratio(col("AMT_CREDIT"), income),
        "ANNUITY_INCOME_RATIO": _ratio(col("AMT_ANNUITY"), income),
        "PAYMENT_RATE": _ratio(col("AMT_ANNUITY"), col("AMT_CREDIT")),
        "CREDIT_GOODS_RATIO": _ratio(col("AMT_CREDIT"), col("AMT_GOODS_PRICE")),
        "INCOME_PER_PERSON": _ratio(income, col("CNT_FAM_MEMBERS")),
        "EMPLOYED_AGE_RATIO": _ratio(col("DAYS_EMPLOYED"), col("DAYS_BIRTH")),
        "EXT_SOURCE_MEAN": ext.mean(axis=1),
        "EXT_SOURCE_MIN": ext.min(axis=1),
        "EXT_SOURCE_MAX": ext.max(axis=1),
        "EXT_SOURCE_STD": ext.std(axis=1),
        "BUREAU_ACTIVE_RATIO": _ratio(col("BUREAU_ACTIVE_COUNT"), col("BUREAU_LOAN_COUNT")),
        "BUREAU_DEBT_CREDIT_RATIO": _ratio(col("BUREAU_AMT_CREDIT_SUM_DEBT"), col("BUREAU_AMT_CREDIT_SUM")),
        "BUREAU_DEBT_INCOME_RATIO": _ratio(col("BUREAU_AMT_CREDIT_SUM_DEBT"), income),
        "PREV_REFUSED_RATIO": _ratio(col("PREV_REFUSED_COUNT"), col("PREV_APP_COUNT")),
        "PREV_APPROVED_RATIO": _ratio(col("PREV_APPROVED_COUNT"), col("PREV_APP_COUNT")),
    }
    derived_frame = pd.DataFrame(derived, index=df.index).astype("float32")
    return pd.concat([df.drop(columns=[c for c in derived if c in df]), derived_frame], axis=1)


# --------------------------------------------------------------------------- #
# Feature specification
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSpec:
    base_features: list[str]
    feature_names: list[str]
    categories: dict[str, list[str]]
    defaults: dict[str, float | str | None]
    missing_rates: dict[str, float]
    groups: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def fit(cls, base: pd.DataFrame) -> "FeatureSpec":
        base_features = [c for c in base.columns if c not in NON_FEATURE_COLUMNS and c not in DERIVED_FEATURES]
        categories = {
            c: sorted(str(v) for v in base[c].dropna().unique())
            for c in base_features
            if c in CATEGORICAL_FEATURES
        }
        missing_rates = base[base_features].isna().mean()
        defaults: dict[str, float | str | None] = {}
        for column in base_features:
            if missing_rates[column] >= MISSING_DEFAULT_THRESHOLD:
                defaults[column] = None
            elif column in categories:
                defaults[column] = str(base[column].mode(dropna=True).iloc[0])
            else:
                defaults[column] = float(base[column].median())
        present = set(base_features)
        return cls(
            base_features=base_features,
            feature_names=base_features + DERIVED_FEATURES,
            categories=categories,
            defaults=defaults,
            missing_rates={c: round(float(v), 4) for c, v in missing_rates.items()},
            groups={g: [c for c in cols if c in present] for g, cols in FEATURE_GROUPS.items()},
        )

    @property
    def numeric_base_features(self) -> list[str]:
        return [c for c in self.base_features if c not in self.categories]

    def transform(self, base: pd.DataFrame) -> pd.DataFrame:
        """Base-feature frame -> model matrix (derived features, dtypes, column order)."""
        df = base.reindex(columns=self.base_features)
        numeric = df[self.numeric_base_features].apply(pd.to_numeric, errors="coerce").astype("float32")
        if "DAYS_EMPLOYED" in numeric:
            numeric["DAYS_EMPLOYED"] = numeric["DAYS_EMPLOYED"].where(numeric["DAYS_EMPLOYED"] != DAYS_EMPLOYED_ANOMALY)
        categorical = pd.DataFrame(
            {column: self._categorical(df[column], vocabulary) for column, vocabulary in self.categories.items()},
            index=df.index,
        )
        return add_derived_features(pd.concat([numeric, categorical], axis=1))[self.feature_names]

    @staticmethod
    def _categorical(values: pd.Series, vocabulary: list[str]) -> pd.Categorical:
        """Fixed-vocabulary categorical; values never seen in training become missing."""
        values = values.astype(object)
        return pd.Categorical(values.where(values.isin(vocabulary), None), categories=vocabulary)

    def frame_from_records(self, records: list[dict]) -> pd.DataFrame:
        """Model matrix from partial base-feature records.

        A feature absent from a record takes its training default; a feature
        present with value None is treated as missing.
        """
        rows = []
        for record in records:
            row = {}
            for column in self.base_features:
                value = record[column] if column in record else self.defaults[column]
                row[column] = np.nan if value is None else value
            rows.append(row)
        return self.transform(pd.DataFrame(rows, columns=self.base_features))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpec":
        return cls(**data)
