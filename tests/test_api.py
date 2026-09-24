import json

from fastapi.testclient import TestClient

from backend.schemas import EXAMPLE_PROFILE


def test_root_and_health(client):
    root = client.get("/")
    assert root.status_code == 200 and root.json()["docs"] == "/docs"
    health = client.get("/api/v1/health").json()
    assert health["status"] == "ok" and health["model_loaded"] and health["model_version"]


def test_model_endpoints(client):
    info = client.get("/api/v1/model").json()
    assert info["metrics"]["test"]["roc_auc"] > 0.5
    assert info["policy"]["approve_below"] < info["policy"]["decline_at"]
    assert [b["code"] for b in info["scorecard"]["bands"]] == ["A", "B", "C", "D", "E"]

    performance = client.get("/api/v1/model/performance").json()
    assert len(performance["roc"]["model"]["fpr"]) > 10
    assert len(client.get("/api/v1/model/importance?top=5").json()["importance"]) == 5

    schema = client.get("/api/v1/schema").json()
    assert "Cash loans" in schema["options"]["contract_type"]
    assert schema["defaults"]["annual_income"] > 0
    assert "annual_income" in schema["profile"]["required"]

    insights = client.get("/api/v1/insights").json()
    assert insights["kpis"]["applicants"] > 0 and insights["segments"][0]["rows"]


def test_score_applicant(client):
    response = client.post("/api/v1/score?top_k=5", json=EXAMPLE_PROFILE)
    assert response.status_code == 200
    body = response.json()
    assert 0 < body["probability_of_default"] < 1
    assert 300 <= body["credit_score"] <= 850
    assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert len(body["explanations"]) <= 5
    assert set(body["key_metrics"]) >= {"CREDIT_INCOME_RATIO", "EXT_SOURCE_MEAN"}
    assert "X-Process-Time-Ms" in response.headers


def test_score_rejects_bad_input(client):
    missing_income = {k: v for k, v in EXAMPLE_PROFILE.items() if k != "annual_income"}
    assert client.post("/api/v1/score", json=missing_income).status_code == 422
    assert client.post("/api/v1/score", json=EXAMPLE_PROFILE | {"age_years": 12}).status_code == 422
    unknown_category = client.post("/api/v1/score", json=EXAMPLE_PROFILE | {"education": "Hogwarts"})
    assert unknown_category.status_code == 422 and "education" in unknown_category.json()["detail"][0]


def test_score_raw_features(client):
    response = client.post("/api/v1/score/features", json={"features": {"EXT_SOURCE_2": 0.7, "NOT_A_FEATURE": 1}})
    assert response.status_code == 200
    body = response.json()
    assert body["warnings"] == ["Ignored unknown features: NOT_A_FEATURE"]
    empty = client.post("/api/v1/score/features", json={"features": {}})
    assert empty.status_code == 200  # every feature falls back to the training default


def test_batch_scoring_reports_row_errors(client):
    rows = [
        EXAMPLE_PROFILE | {"applicant_id": "A-1"},
        EXAMPLE_PROFILE | {"applicant_id": "A-2", "annual_income": None},
        EXAMPLE_PROFILE | {"applicant_id": 3, "ext_source_1": float("nan"), "occupation": ""},
        {"applicant_id": "A-4", "age_years": "not a number"},
    ]
    # json.dumps emits a bare NaN literal, as Python's `requests` does; httpx would refuse to encode it.
    response = client.post("/api/v1/score/batch", content=json.dumps({"applicants": rows}),
                           headers={"content-type": "application/json"})
    assert response.status_code == 200
    body = response.json()
    summary, results = body["summary"], body["results"]
    assert summary["submitted"] == 4 and summary["scored"] == 2 and summary["failed"] == 2
    assert sum(summary["decisions"].values()) == 2
    assert results[0]["applicant_id"] == "A-1" and results[0]["result"]["reasons"] is not None
    assert results[0]["result"]["explanations"] == []
    assert "annual_income" in results[1]["error"]
    assert results[2]["applicant_id"] == "3" and results[2]["error"] is None
    assert results[3]["result"] is None and results[3]["error"]


def test_api_without_model_degrades_gracefully(tmp_path, monkeypatch):
    from backend.app import create_app

    monkeypatch.setenv("CREDSCORE_MODEL_DIR", str(tmp_path))
    with TestClient(create_app()) as client:
        health = client.get("/api/v1/health").json()
        assert health["status"] == "degraded" and not health["model_loaded"]
        response = client.post("/api/v1/score", json=EXAMPLE_PROFILE)
        assert response.status_code == 503 and "credscore.pipeline" in response.json()["detail"]
