import json

import pytest
from pydantic import ValidationError

from backend.schemas import EXAMPLE_PROFILE
from credscore import config
from credscore.model import CreditModel, ModelNotFoundError
from credscore.profile import ApplicantProfile, profile_to_base_features


def test_pipeline_writes_a_complete_bundle(model_dir, data_dir):
    for name in (config.MODEL_FILE, config.METADATA_FILE, config.REPORT_FILE, config.INSIGHTS_FILE):
        assert (model_dir / name).exists(), name
    assert (data_dir / config.FEATURES_FILE).exists()
    metadata = json.loads((model_dir / config.METADATA_FILE).read_text())
    assert metadata["metrics"]["test"]["roc_auc"] > 0.6  # synthetic data carries real signal
    policy = metadata["policy"]
    assert 0 < policy["approve_below"] < policy["decline_at"] < 1
    report = json.loads((model_dir / config.REPORT_FILE).read_text())
    assert {"roc", "calibration", "gains", "bands", "policy", "importance", "fairness"} <= set(report)
    top_features = [row["feature"] for row in report["importance"][:5]]
    assert any(f.startswith("EXT_SOURCE") for f in top_features), top_features
    insights = json.loads((model_dir / config.INSIGHTS_FILE).read_text())
    assert insights["kpis"]["applicants"] == 3000 and len(insights["segments"]) >= 10


def test_missing_bundle_raises_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="credscore.pipeline"):
        CreditModel.load(tmp_path)


def test_profile_validation():
    with pytest.raises(ValidationError):
        ApplicantProfile(annual_income=1, credit_amount=1, annuity_amount=1, bureau_loans=1, bureau_active_loans=2)
    with pytest.raises(ValidationError):
        ApplicantProfile(annual_income=1, credit_amount=1, annuity_amount=1, unknown_field=3)
    with pytest.raises(ValidationError):
        ApplicantProfile(annual_income=-5, credit_amount=1, annuity_amount=1)


def test_profile_mapping_encodes_history_and_employment():
    new_client = ApplicantProfile(annual_income=100_000, credit_amount=300_000, annuity_amount=15_000,
                                  income_type="Pensioner", age_years=65)
    features = profile_to_base_features(new_client)
    assert features["DAYS_BIRTH"] == pytest.approx(-65 * 365.25)
    assert features["DAYS_EMPLOYED"] is None and features["ORGANIZATION_TYPE"] == "XNA"
    assert features["BUREAU_LOAN_COUNT"] is None and features["PREV_APP_COUNT"] is None
    assert features["INSTAL_COUNT"] is None and features["OWN_CAR_AGE"] is None
    assert features["AMT_GOODS_PRICE"] == 300_000

    repeat = ApplicantProfile(**EXAMPLE_PROFILE)
    features = profile_to_base_features(repeat)
    assert features["BUREAU_LOAN_COUNT"] == 4 and features["PREV_APPROVED_COUNT"] == 2
    assert features["BUREAU_AMT_CREDIT_SUM"] == 850_000
    assert features["DAYS_EMPLOYED"] == pytest.approx(-7.5 * 365.25)
    assert "ORGANIZATION_TYPE" not in features  # unknown employer type -> training default

    # Loans on record but no amounts given: fall back to the training default
    # rather than claiming this applicant borrowed nothing.
    partial = ApplicantProfile(**EXAMPLE_PROFILE | {"bureau_total_credit": 0})
    assert "BUREAU_AMT_CREDIT_SUM" not in profile_to_base_features(partial)


def test_scores_are_consistent_and_explained(service):
    strong = ApplicantProfile(**EXAMPLE_PROFILE | {"ext_source_2": 0.8, "ext_source_3": 0.85})
    weak = ApplicantProfile(**EXAMPLE_PROFILE | {"ext_source_2": 0.05, "ext_source_3": 0.05})
    good, bad = service.score_profiles([strong, weak])
    assert good["probability_of_default"] < bad["probability_of_default"]
    assert good["credit_score"] > bad["credit_score"]
    for result in (good, bad):
        assert result["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
        assert result["risk_band"]["min_score"] <= result["credit_score"] <= result["risk_band"]["max_score"]
        assert 0 < len(result["explanations"]) <= 8
        assert {e["source"] for e in result["explanations"]} <= {"applicant", "default", "derived"}
    assert any(e["feature"].startswith("EXT_SOURCE") for e in bad["explanations"])
    assert bad["reasons"], "a weak applicant should get risk reasons"


def test_contributions_add_up_to_the_prediction(service):
    import numpy as np

    X = service.spec.frame_from_records([profile_to_base_features(ApplicantProfile(**EXAMPLE_PROFILE))])
    contributions = service.model.contributions(X)
    pd_value = service.model.predict_pd(X)[0]
    assert 1 / (1 + np.exp(-contributions.sum())) == pytest.approx(pd_value, rel=1e-4)


def test_profile_errors_flag_unknown_categories(service):
    profile = ApplicantProfile(**EXAMPLE_PROFILE | {"education": "Hogwarts"})
    errors = service.profile_errors(profile)
    assert len(errors) == 1 and errors[0].startswith("education")


def test_policy_can_be_overridden_by_environment(model_dir, monkeypatch):
    from credscore.service import ScoringService

    monkeypatch.setenv("CREDSCORE_APPROVE_PD", "0.01")
    monkeypatch.setenv("CREDSCORE_DECLINE_PD", "0.5")
    service = ScoringService.load(model_dir)
    assert service.policy.approve_below == 0.01 and service.policy.decline_at == 0.5
