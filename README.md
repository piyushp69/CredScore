# CredScore

Explainable credit default risk scoring on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset: a reproducible training pipeline, a FastAPI scoring service and a Streamlit underwriting dashboard.

Every applicant gets a probability of default, a 300-850 credit score, a risk band, a recommended decision and a per-feature explanation of how that score was reached.

```
dataset/*.csv ──► credscore.pipeline ──► models/          ──┬─► dashboard/ (Streamlit)   underwriting · batch scoring
                  features · train        model.ubj          │                           portfolio · model performance
                  evaluate · insights     metadata.json      │
                                          report.json        └─► backend/ (FastAPI)      /api/v1/score · /api/v1/model
                                          insights.json                                  /api/v1/insights ...
```

The dashboard and the API are independent: both load the model bundle in-process through
`credscore.service.ScoringService`, so the dashboard needs no API server running.

## Results

Measured on a held-out test set of 46,127 applications that the model never saw during training or tuning:

| Metric | Model | Baseline (external scores only) |
|---|---|---|
| ROC AUC | **0.784** | 0.718 |
| Gini | 0.567 | 0.435 |
| KS | 0.426 | 0.326 |
| PR AUC | 0.275 | 0.195 |
| Brier score | 0.0662 | not calibrated |

Predicted probabilities match reality: across risk deciles, mean predicted PD tracks the observed default rate (e.g. 30.3% predicted vs 29.8% observed in the riskiest decile), and the average predicted PD of 7.97% matches the 8.07% actual base rate.

The decision policy derived from the validation set approves the safest 70% and declines the riskiest 10%:

| Decision | Applicants | Default rate | Share of all defaulters |
|---|---|---|---|
| Approve (PD < 8.4%) | 70.1% | 3.6% | 31.1% |
| Manual review | 19.8% | 12.9% | 31.7% |
| Decline (PD >= 18.9%) | 10.1% | 29.8% | 37.2% |

## Quick start

```bash
pip install -r requirements.txt

# 1. Data + model (downloads ~1.5 GB from Kaggle if dataset/ is empty)
python -m credscore.pipeline all          # ~2 minutes with a GPU, ~10 on CPU

# 2. Dashboard -> http://localhost:8501
streamlit run streamlit_app.py

# 3. (optional) REST API -> http://127.0.0.1:8000  (docs at /docs)
uvicorn backend.app:app --port 8000
```

The Kaggle download needs credentials (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`) and acceptance of the competition rules. If the CSVs are already in `dataset/`, the download step is skipped.

Individual steps: `python -m credscore.pipeline {download,features,train,insights}`, with `--sample 20000` for a fast run, `--device cpu|cuda|auto`, `--data-dir`, `--model-dir`.

To serve an already-trained bundle in containers: `docker compose up --build` (dashboard on 8501, API on 8000, both from one image). Training still runs on the host, and `./models` is mounted read-only.

## Dashboard

Streamlit app (`streamlit_app.py` + `dashboard/`) with interactive Plotly charts on a colour-vision-safe
palette. Theme and upload limits live in `.streamlit/config.toml`.

| Page | What it does |
|---|---|
| **Underwriting** | Score one applicant from a form (with example presets), see the decision, the SHAP factor breakdown, plain-language risk reasons, and a what-if curve that varies one input while holding the rest fixed. Downloads a JSON decision report. |
| **Batch scoring** | Score a CSV of applicants (template provided, or 250 demo applicants). Score distribution, decision mix, per-row reasons and errors, results download. |
| **Portfolio insights** | Observed default rates across 15 applicant segments of the 307,511 labelled applications, plus where risk concentrates. |
| **Model performance** | Discrimination, calibration, decision-policy economics, global feature importance and fairness monitoring, all from the held-out test set. |

The model is loaded once per server process (`st.cache_resource`) and shared by every session.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/health` | Liveness and loaded model version |
| `GET /api/v1/model` | Version, test metrics, decision policy, scorecard |
| `GET /api/v1/model/performance` | Curves, calibration, gains, bands, fairness |
| `GET /api/v1/model/importance` | Global mean \|SHAP\| importance |
| `GET /api/v1/schema` | Applicant fields, allowed category values, defaults |
| `GET /api/v1/insights` | Portfolio default rates by segment |
| `POST /api/v1/score` | Score one applicant profile |
| `POST /api/v1/score/batch` | Score up to 10,000 profiles; bad rows are reported, not fatal |
| `POST /api/v1/score/features` | Score raw model features (advanced) |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/score \
  -H 'content-type: application/json' \
  -d '{"age_years":38,"annual_income":225000,"credit_amount":600000,"annuity_amount":28000,
       "education":"Higher education","ext_source_2":0.68,"ext_source_3":0.62,
       "bureau_loans":4,"bureau_active_loans":1,"bureau_total_credit":850000,
       "prev_applications":2,"prev_approved":2}'
```

```json
{
  "probability_of_default": 0.021296,
  "credit_score": 698,
  "risk_band": {"code": "B", "label": "Low risk", "min_score": 660, "max_score": 719, "status": "good"},
  "decision": "APPROVE",
  "decision_reason": "PD 2.1% is below the approval cut-off of 8.4%.",
  "explanations": [
    {"feature": "EXT_SOURCE_MEAN", "label": "External scores (average)", "value": 0.65,
     "display_value": "0.650", "contribution": -0.56954, "effect": "decreases_risk", "source": "derived"}
  ],
  "reasons": ["Age: 38 years", "External score 1: not available", "Car age (years): not available"]
}
```

Applicant fields use human units (years, amounts, counts); the service maps them onto model features. Unspecified optional fields fall back to the training population's typical value, and the response marks which factors came from the applicant (`source: "applicant"`) versus a default.

## Modelling decisions

- **Split before anything else.** Train/validation/test (70/15/15, stratified) is decided up front. Every reported number comes from the test set; the validation set only drives early stopping and the policy cut-offs.
- **No oversampling.** The 8% default rate is left alone. SMOTE on this data interpolates one-hot flags and median-imputed columns into values that cannot occur (a fractional `FLAG_PHONE`), so a model learns to detect synthetic rows instead of risk, and the resulting probabilities are no longer calibrated — which the credit score and the cut-offs depend on.
- **Missing stays missing.** XGBoost learns a default direction per split, so no imputer is needed and "no credit bureau history" keeps its meaning instead of becoming a median.
- **Native categorical splits.** Category vocabularies are stored in the model bundle, so no encoder has to be replicated at serving time and unseen categories degrade to missing.
- **Explanations from the model itself.** Exact TreeSHAP via XGBoost's `pred_contribs`, so there is no separate explainer artifact to drift or unpickle.
- **Protected attributes.** `CODE_GENDER` is excluded from the model and retained only for fairness reporting. Application-process artifacts (weekday/hour of application) and near-constant flags are dropped too.
- **One feature module.** Training and serving share `credscore/features.py`; derived ratios are always recomputed and never accepted as input, so the two cannot drift.
- **Native model format.** `model.ubj` rather than a pickle: portable across versions and safe to load.

Scoring uses a standard scorecard scaling: 650 points = 20:1 good:bad odds, and every 40 points doubles the odds, clipped to 300-850.

## Layout

```
credscore/           core package (shared by pipeline and API)
  features.py        feature engineering + FeatureSpec (vocabularies, serving defaults)
  profile.py         human-friendly applicant profile -> model features
  model.py           model bundle: booster + spec + metadata
  service.py         scoring, explanations, decisions
  scoring.py         PD -> credit score, risk bands, decision policy
  evaluation.py      metrics, curves, gains, fairness
  labels.py          human-readable feature names and value formatting
  pipeline/          download -> features -> train -> insights (python -m credscore.pipeline)
backend/             FastAPI service (app factory, schemas)
streamlit_app.py     dashboard entrypoint: navigation, model status
dashboard/           Streamlit pages
  common.py          model loading, formatting, chart styling, example applicants
  underwriting.py    · batch.py · portfolio.py · performance.py
tests/               pytest suite incl. synthetic-data pipeline fixture
```

## Tests

```bash
python -m pytest            # 49 tests, ~25 s
```

The suite builds a synthetic Home Credit-shaped dataset, runs the real pipeline CLI over it, then exercises the API and every dashboard page (headlessly, via Streamlit's `AppTest`) against that model — so it needs neither the 1.5 GB dataset nor a trained model.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CREDSCORE_DATA_DIR` | `dataset/` | Raw CSVs and `features.parquet` |
| `CREDSCORE_MODEL_DIR` | `models/` | Model bundle, report, insights |
| `CREDSCORE_APPROVE_PD` / `CREDSCORE_DECLINE_PD` | from training | Override decision cut-offs at serving time |
| `CREDSCORE_CORS_ORIGINS` | `*` | Comma-separated CORS allow-list |

## Limitations

- Trained on Home Credit's 2018 competition sample; it is a portfolio demonstrator, not a production credit policy.
- `bureau_balance.csv`, `POS_CASH_balance.csv` and `credit_card_balance.csv` are not used; adding them is the most obvious route past AUC 0.784.
- Decision cut-offs are set by target approval rates, not by expected loss or profit — real policy needs pricing, exposure and regulatory input.
- Fairness reporting covers gender and age bands on the test set only; it is monitoring, not a compliance review.
