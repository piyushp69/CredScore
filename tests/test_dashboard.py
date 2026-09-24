"""Streamlit dashboard: every page renders and the main interactions work, headlessly via AppTest."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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


def test_portfolio_and_performance_pages_render():
    at = page("portfolio").run()
    assert_clean(at)
    assert at.selectbox[0].options  # segment breakdowns available
    at.selectbox[0].select_index(len(at.selectbox[0].options) - 1).run()
    assert_clean(at)

    at = page("performance").run()
    assert_clean(at)
    assert any(m.label == "ROC AUC" for m in at.metric)


def test_pages_explain_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDSCORE_MODEL_DIR", str(tmp_path))
    at = page("performance").run()
    assert not at.exception
    assert "credscore.pipeline" in at.error[0].value
