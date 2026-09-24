"""Streamlit dashboard: every page renders and the main interactions work, headlessly via AppTest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from credscore import config
from dashboard.common import RISKY, pct

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 120


@pytest.fixture(autouse=True)
def dashboard_model(model_dir, monkeypatch):
    from dashboard.common import _load_service

    monkeypatch.setenv("CREDSCORE_MODEL_DIR", str(model_dir))
    _load_service.clear()
    yield
    _load_service.clear()


def page(module: str) -> AppTest:
    return AppTest.from_string(f"from dashboard import {module}\n{module}.render()", default_timeout=TIMEOUT)


def button(at: AppTest, text: str):
    return next(b for b in at.button if text in str(b.label))


def assert_clean(at: AppTest) -> None:
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]


def test_app_shell_loads_underwriting_by_default():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=TIMEOUT).run()
    assert_clean(at)
    assert at.title[0].value == "Applicant underwriting"
    assert any("Model loaded" in s.value for s in at.success)


def test_underwriting_scores_presets_and_builds_what_if():
    at = page("underwriting").run()
    assert_clean(at)
    button(at, "Score applicant").click().run()
    assert_clean(at)
    labels = [m.label for m in at.metric]
    assert "Credit score" in labels and "Probability of default" in labels
    typical = int(next(m.value for m in at.metric if m.label == "Credit score"))

    button(at, "Risky applicant").click().run()
    button(at, "Score applicant").click().run()
    assert_clean(at)
    risky = int(next(m.value for m in at.metric if m.label == "Credit score"))
    assert risky < typical
    assert at.selectbox(key="uw_whatif").value  # what-if curve rendered below the result

    at.selectbox(key="uw_whatif").select("annual_income").run()
    assert_clean(at)


def test_underwriting_inputs_survive_switching_pages():
    # Streamlit drops the state of widgets a run does not draw, as happens while another page is open.
    at = AppTest.from_string(
        "import streamlit as st\nfrom dashboard import underwriting\n"
        "if not st.session_state.get('elsewhere'):\n    underwriting.render()",
        default_timeout=TIMEOUT,
    ).run()
    button(at, "Risky applicant").click().run()
    button(at, "Score applicant").click().run()
    at.session_state["elsewhere"] = True
    at.run()
    at.session_state["elsewhere"] = False
    at.run()
    assert_clean(at)
    assert at.number_input(key="uw_age_years").value == RISKY["age_years"]  # the applicant still on screen


def test_underwriting_optional_inputs_can_be_cleared():
    at = page("underwriting").run()
    button(at, "Strong applicant").click().run()
    at.number_input(key="uw_ext_source_1").set_value(None)
    at.number_input(key="uw_goods_price").set_value(None)
    button(at, "Score applicant").click().run()
    assert_clean(at)
    profile = at.session_state["uw_scored"]["profile"]
    assert profile["ext_source_1"] is None and profile["goods_price"] is None  # not provided, not the minimum


def test_what_if_sweep_spans_the_applicants_value():
    from dashboard.underwriting import _sweep

    assert _sweep("age_years", 75.0)[-1] == 75  # beyond the usual 20-70 span
    small = _sweep("credit_amount", 5000.0)  # below the usual 20,000 floor
    assert small == sorted(small) and small[0] <= 5000 <= small[-1]
    assert _sweep("ext_source_2", None)[0] == 0.02  # not provided: the usual span


def test_underwriting_reports_invalid_profile():
    at = page("underwriting").run()
    at.number_input(key="uw_bureau_loans").set_value(1)
    at.number_input(key="uw_bureau_active_loans").set_value(3)
    button(at, "Score applicant").click().run()
    assert any("bureau_active_loans cannot exceed" in e.value for e in at.error)


def test_batch_scores_demo_applicants():
    at = page("batch").run()
    assert_clean(at)
    button(at, "Load 250 demo applicants").click().run()
    button(at, "Score 250 applicants").click().run()
    assert_clean(at)
    assert next(m.value for m in at.metric if m.label == "Scored") == "250"
    assert len(at.dataframe[0].value) == 250


def test_batch_reports_csv_without_rows():
    at = page("batch").run()
    at.file_uploader[0].upload("empty.csv", b"applicant_id,annual_income,credit_amount,annuity_amount\n",
                               "text/csv").run()
    assert any("no applicant rows" in e.value for e in at.error)


def test_portfolio_and_performance_pages_render():
    at = page("portfolio").run()
    assert_clean(at)
    assert at.selectbox[0].options  # segment breakdowns available
    at.selectbox[0].select_index(len(at.selectbox[0].options) - 1).run()
    assert_clean(at)

    at = page("performance").run()
    assert_clean(at)
    assert any(m.label == "ROC AUC" for m in at.metric)


def test_policy_outcomes_keep_the_cutoffs_they_were_measured_at(model_dir, monkeypatch):
    monkeypatch.setenv("CREDSCORE_APPROVE_PD", "0.01")
    monkeypatch.setenv("CREDSCORE_DECLINE_PD", "0.9")
    trained = json.loads((model_dir / config.REPORT_FILE).read_text())["policy"]["cutoffs"]
    at = page("performance").run()
    assert_clean(at)
    policy = next(i.value for i in at.info if "manual review" in i.value)
    assert pct(trained["approve_below"]) in policy and pct(trained["decline_at"]) in policy
    assert any(pct(0.01) in w.value and pct(0.9) in w.value for w in at.warning)


def test_pages_explain_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDSCORE_MODEL_DIR", str(tmp_path))
    at = page("performance").run()
    assert not at.exception
    assert "credscore.pipeline" in at.error[0].value
