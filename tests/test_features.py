import numpy as np
import pandas as pd
import pytest

from credscore.features import (
    DAYS_EMPLOYED_ANOMALY,
    DERIVED_FEATURES,
    FeatureSpec,
    add_derived_features,
    aggregate_bureau,
    aggregate_installments,
    aggregate_previous,
    build_base_table,
    clean_application,
    is_model_application_column,
)


def test_clean_application_fixes_known_quirks():
    app = pd.DataFrame({
        "FLAG_OWN_CAR": ["Y", "N"], "FLAG_OWN_REALTY": ["N", "Y"],
        "DAYS_EMPLOYED": [DAYS_EMPLOYED_ANOMALY, -400], "CODE_GENDER": ["XNA", "F"],
    })
    cleaned = clean_application(app)
    assert cleaned["FLAG_OWN_CAR"].tolist() == [1.0, 0.0]
    assert np.isnan(cleaned.loc[0, "DAYS_EMPLOYED"]) and cleaned.loc[1, "DAYS_EMPLOYED"] == -400
    assert pd.isna(cleaned.loc[0, "CODE_GENDER"])


@pytest.mark.parametrize("column, kept", [
    ("CODE_GENDER", False), ("TARGET", False), ("WEEKDAY_APPR_PROCESS_START", False),
    ("FLAG_DOCUMENT_2", False), ("FLAG_DOCUMENT_3", True), ("APARTMENTS_MEDI", False),
    ("APARTMENTS_MODE", False), ("APARTMENTS_AVG", True), ("TOTALAREA_MODE", True), ("HOUSETYPE_MODE", True),
    ("EXT_SOURCE_2", True),
])
def test_application_column_selection(column, kept):
    assert is_model_application_column(column) is kept


def test_bureau_aggregation():
    bureau = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2], "SK_ID_BUREAU": [10, 11, 12], "CREDIT_ACTIVE": ["Active", "Closed", "Closed"],
        "DAYS_CREDIT": [-100, -500, -50], "CREDIT_DAY_OVERDUE": [0, 5, 0], "DAYS_CREDIT_ENDDATE": [100, -200, 10],
        "AMT_CREDIT_MAX_OVERDUE": [np.nan, 300.0, np.nan], "CNT_CREDIT_PROLONG": [0, 1, 0],
        "AMT_CREDIT_SUM": [1000.0, 3000.0, 500.0], "AMT_CREDIT_SUM_DEBT": [400.0, 0.0, 0.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 0.0],
    })
    agg = aggregate_bureau(bureau)
    assert agg.loc[1, "BUREAU_LOAN_COUNT"] == 2 and agg.loc[1, "BUREAU_ACTIVE_COUNT"] == 1
    assert agg.loc[1, "BUREAU_DAYS_CREDIT_MAX"] == -100
    assert agg.loc[1, "BUREAU_AMT_CREDIT_SUM_DEBT"] == 400
    assert agg.loc[1, "BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX"] == 300
    assert np.isnan(agg.loc[2, "BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX"])


def test_previous_and_installment_aggregation():
    prev = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1], "SK_ID_PREV": [1, 2, 3], "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Canceled"],
        "AMT_APPLICATION": [100.0, 200.0, 50.0], "AMT_CREDIT": [100.0, 0.0, 50.0], "AMT_ANNUITY": [10.0, 20.0, 5.0],
        "DAYS_DECISION": [-10, -20, -30], "CNT_PAYMENT": [12.0, 24.0, 6.0],
    })
    agg = aggregate_previous(prev)
    assert agg.loc[1, ["PREV_APP_COUNT", "PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT"]].tolist() == [3, 1, 1]
    assert agg.loc[1, "PREV_APP_CREDIT_RATIO_MEAN"] == pytest.approx(1.0)  # zero credit is excluded

    inst = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1, 1], "DAYS_INSTALMENT": [-100, -70, -40, -10],
        "DAYS_ENTRY_PAYMENT": [-105, -60, -40, np.nan], "AMT_INSTALMENT": [100.0, 100.0, 100.0, 100.0],
        "AMT_PAYMENT": [100.0, 50.0, 100.0, np.nan],
    })
    agg = aggregate_installments(inst)
    assert agg.loc[1, "INSTAL_COUNT"] == 4
    assert agg.loc[1, "INSTAL_LATE_SHARE"] == pytest.approx(1 / 3)  # unpaid installment is not counted
    assert agg.loc[1, "INSTAL_DPD_MAX"] == 10
    assert agg.loc[1, "INSTAL_PAYMENT_RATIO_MEAN"] == pytest.approx(2.5 / 3)


def test_derived_features_handle_zero_and_missing():
    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [100.0, 0.0], "AMT_CREDIT": [300.0, 300.0], "AMT_ANNUITY": [30.0, 30.0],
        "AMT_GOODS_PRICE": [250.0, np.nan], "CNT_FAM_MEMBERS": [2.0, 1.0], "DAYS_EMPLOYED": [-365.0, np.nan],
        "DAYS_BIRTH": [-3650.0, -3650.0], "EXT_SOURCE_1": [np.nan, np.nan], "EXT_SOURCE_2": [0.2, np.nan],
        "EXT_SOURCE_3": [0.6, np.nan],
    })
    out = add_derived_features(df)
    assert set(DERIVED_FEATURES) <= set(out.columns)
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(3.0)
    assert out.loc[0, "PAYMENT_RATE"] == pytest.approx(0.1)
    assert out.loc[0, "EXT_SOURCE_MEAN"] == pytest.approx(0.4)
    assert np.isnan(out.loc[1, "CREDIT_INCOME_RATIO"])  # zero income -> missing, not inf
    assert np.isnan(out.loc[1, "EXT_SOURCE_MEAN"])
    assert np.isnan(out.loc[0, "BUREAU_ACTIVE_RATIO"])  # absent inputs -> missing


def test_base_table_and_spec(raw_tables):
    base = build_base_table(**raw_tables)
    assert base["SK_ID_CURR"].is_unique and len(base) == len(raw_tables["application"])
    assert "CODE_GENDER" in base and "WEEKDAY_APPR_PROCESS_START" not in base
    has_bureau = base["SK_ID_CURR"].isin(raw_tables["bureau"]["SK_ID_CURR"])
    assert base.loc[~has_bureau, "BUREAU_LOAN_COUNT"].isna().all()

    spec = FeatureSpec.fit(base)
    assert "CODE_GENDER" not in spec.feature_names and "TARGET" not in spec.feature_names
    assert spec.feature_names[-len(DERIVED_FEATURES):] == DERIVED_FEATURES
    assert spec.defaults["EXT_SOURCE_1"] is None  # ~50% missing -> defaults to missing
    assert isinstance(spec.defaults["AMT_INCOME_TOTAL"], float)
    assert spec.defaults["NAME_CONTRACT_TYPE"] == "Cash loans"

    X = spec.transform(base.head(50))
    assert list(X.columns) == spec.feature_names
    assert isinstance(X["NAME_CONTRACT_TYPE"].dtype, pd.CategoricalDtype)
    assert list(X["NAME_CONTRACT_TYPE"].cat.categories) == spec.categories["NAME_CONTRACT_TYPE"]

    restored = FeatureSpec.from_dict(spec.to_dict())
    pd.testing.assert_frame_equal(restored.transform(base.head(50)), X)


def test_frame_from_records_defaults_and_unknown_categories(raw_tables):
    spec = FeatureSpec.fit(build_base_table(**raw_tables))
    X = spec.frame_from_records([
        {},
        {"AMT_INCOME_TOTAL": 1000.0, "EXT_SOURCE_2": None, "NAME_CONTRACT_TYPE": "Never seen",
         "DAYS_EMPLOYED": DAYS_EMPLOYED_ANOMALY},
    ])
    assert X.loc[0, "AMT_INCOME_TOTAL"] == pytest.approx(spec.defaults["AMT_INCOME_TOTAL"], rel=1e-6)
    assert X.loc[1, "AMT_INCOME_TOTAL"] == 1000.0
    assert np.isnan(X.loc[1, "EXT_SOURCE_2"])
    assert pd.isna(X.loc[1, "NAME_CONTRACT_TYPE"])
    assert np.isnan(X.loc[1, "DAYS_EMPLOYED"])
