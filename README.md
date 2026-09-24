<div align="center">

# 💳 CredScore

**Explainable credit risk scoring for loan applications — from raw data to a decision you can defend.**

Every applicant gets a probability of default, a 300–850 credit score, a risk band, a recommended decision
and a per-feature explanation of how that score was reached.

[![Open the live app](https://img.shields.io/badge/Open%20the%20live%20app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://credscore.streamlit.app)
[![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in%20Codespaces-24292F?style=for-the-badge&logo=github&logoColor=white)](https://codespaces.new/piyushp69/CredScore?quickstart=1)

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit 1.52+](https://img.shields.io/badge/Streamlit-1.52%2B-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.x-EB5B2D)
![ROC AUC 0.784](https://img.shields.io/badge/ROC%20AUC-0.784-2EA44F)

**[Live demo](https://credscore.streamlit.app)** · [Links](#links) · [Screenshots](#screenshots) · [Results](#results) ·
[Quick start](#quick-start) · [REST API](#rest-api) · [How it works](#how-it-works)

</div>

<br>

<img src="docs/screenshots/underwriting-result.jpg" alt="CredScore scoring a risky applicant: credit score 477, band E, 49.9% probability of default and a decline, with the factors behind it" width="100%">

## Highlights

- **A decision, not just a number** — probability of default, 300–850 credit score, risk band A–E and an
  approve / manual review / decline recommendation from validation-set cut-offs.
- **Explained factor by factor** — exact TreeSHAP contributions from XGBoost itself, plus plain-language risk reasons.
- **What-if analysis** — vary one input and watch the probability of default cross the decision cut-offs.
- **Batch scoring** — score a CSV of applicants; bad rows are reported, not fatal.
- **Honest evaluation** — every metric comes from a held-out test set of 46,127 applications, with calibration,
  policy economics and fairness monitoring.
- **Two front doors** — a Streamlit dashboard and a FastAPI REST service share one scoring engine.

## Links

- 🌐 **Live dashboard:** [credscore.streamlit.app](https://credscore.streamlit.app)
- 📄 **Pages:** [Underwriting](https://credscore.streamlit.app) · [Batch scoring](https://credscore.streamlit.app/batch) ·
  [Portfolio insights](https://credscore.streamlit.app/portfolio) · [Model performance](https://credscore.streamlit.app/performance)
- 💻 **Source code:** [github.com/piyushp69/CredScore](https://github.com/piyushp69/CredScore)
- ☁️ **Try it in your browser:** [Open in GitHub Codespaces](https://codespaces.new/piyushp69/CredScore?quickstart=1)
- 🗂️ **Dataset:** [Home Credit Default Risk on Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)
- 🖥️ **Run locally:** dashboard at [localhost:8501](http://localhost:8501) ·
  API docs at [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) ([ReDoc](http://127.0.0.1:8000/redoc)) —
  see [Quick start](#quick-start)

## Screenshots

### Underwriting

Score one applicant from a form (or start from a typical, strong or risky example), see why the model decided
what it did, and test what would change it.

<img src="docs/screenshots/underwriting.jpg" alt="Underwriting page: applicant form with example presets" width="100%">

<details>
<summary><b>Scored result, explanation and what-if analysis</b> — 3 more screenshots</summary>
<br>

**Scored applicant** — score, band, probability of default, decision and the key affordability ratios.

<img src="docs/screenshots/underwriting-result.jpg" alt="Scored applicant: credit score 477, band E, 49.9% probability of default, decline" width="100%">

**What drove the score** — SHAP contribution of each factor, the main risk factors and the decision policy.

<img src="docs/screenshots/underwriting-explanation.jpg" alt="SHAP factor chart, main risk factors and decision policy" width="100%">

**What-if analysis** — one input varied, everything else fixed, against the approve and decline cut-offs.

<img src="docs/screenshots/underwriting-what-if.jpg" alt="What-if curve of probability of default against external score 2, crossing both cut-offs" width="100%">

</details>

### Batch scoring

Score a whole CSV of applicants (template or 250 demo applicants provided): score distribution, decision mix,
per-row risk reasons and errors, and a results download.

<img src="docs/screenshots/batch-scoring.jpg" alt="Batch scoring of 250 applicants: decision mix, score distribution and decisions chart" width="100%">

<details>
<summary><b>Per-applicant results</b> — 1 more screenshot</summary>
<br>

<img src="docs/screenshots/batch-results.jpg" alt="Results table with score, probability of default, band, decision and main risk factors per applicant" width="100%">

</details>

### Portfolio insights

Observed default rates across 15 segments of the 307,511 labelled applications, and where the risk concentrates.

<img src="docs/screenshots/portfolio.jpg" alt="Portfolio insights: headline figures and default rate by annual income" width="100%">

<details>
<summary><b>Where the risk concentrates</b> — 1 more screenshot</summary>
<br>

<img src="docs/screenshots/portfolio-risk.jpg" alt="Segments with the highest and lowest default rates" width="100%">

</details>

### Model performance

Discrimination, calibration, decision-policy economics, global feature importance and fairness monitoring — all
measured on the held-out test set.

<img src="docs/screenshots/performance.jpg" alt="Model performance: ROC AUC 0.784, Gini, KS, ROC curve and score distributions" width="100%">

<details>
<summary><b>Calibration, decision policy, explainability and fairness</b> — 4 more screenshots</summary>
<br>

**Calibration** — predicted probability of default against observed default rate.

<img src="docs/screenshots/performance-calibration.jpg" alt="Calibration curve and observed default rate by risk band" width="100%">

**Decision policy** — outcomes of the approve / review / decline cut-offs, cumulative gains and risk deciles.

<img src="docs/screenshots/performance-policy.jpg" alt="Decision policy outcomes, cumulative gains chart and risk decile table" width="100%">

**Explainability** — the 20 features that move scores the most, and the inputs deliberately left out.

<img src="docs/screenshots/performance-explainability.jpg" alt="Top 20 features by mean absolute SHAP value and excluded inputs" width="100%">

**Fairness** — approval and default rates by gender and age band; gender is not a model input.

<img src="docs/screenshots/performance-fairness.jpg" alt="Approval rate and observed default rate by gender and by age band" width="100%">

</details>

### REST API docs

Interactive OpenAPI docs at `/docs` for every endpoint of the [REST API](#rest-api).

<img src="docs/screenshots/api-docs.jpg" alt="CredScore API Swagger UI listing the endpoints" width="100%">

## Results

Measured on a held-out test set of **46,127 applications** that the model never saw during training or tuning:

| Metric | Model | Baseline (external scores only) |
|---|---|---|
| ROC AUC | **0.784** | 0.718 |
| Gini | 0.567 | 0.435 |
| KS | 0.426 | 0.326 |
| PR AUC | 0.275 | 0.195 |
| Brier score | 0.0662 | not calibrated |

Predicted probabilities match reality: across risk deciles, mean predicted PD tracks the observed default rate
(e.g. 30.3% predicted vs 29.8% observed in the riskiest decile), and the average predicted PD of 7.97% matches
the 8.07% actual base rate.

The decision policy derived from the validation set approves the safest 70% (PD below 8.4%) and declines the
riskiest 10% (PD of 18.9% or more):

| Decision (share of applicants) | Default rate | Share of all defaulters |
|---|---|---|
| Approve (70.1%) | 3.6% | 31.1% |
| Manual review (19.8%) | 12.9% | 31.7% |
| Decline (10.1%) | 29.8% | 37.2% |

## Quick start

```bash
git clone https://github.com/piyushp69/CredScore.git
cd CredScore
pip install -r requirements.txt

# 1. (optional) Retrain: a trained bundle is committed in models/.
#    Downloads ~1.5 GB from Kaggle if dataset/ is empty.
python -m credscore.pipeline all          # ~2 minutes with a GPU, ~10 on CPU

# 2. Dashboard -> http://localhost:8501
streamlit run streamlit_app.py

# 3. (optional) REST API -> http://127.0.0.1:8000  (docs at /docs)
uvicorn backend.app:app --port 8000
```

The Kaggle download needs credentials (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`) and
acceptance of the [competition rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules).
If the CSVs are already in `dataset/`, the download step is skipped.

Individual steps: `python -m credscore.pipeline {download,features,train,insights}`, with `--sample 20000` for a
fast run, `--device cpu|cuda|auto`, `--data-dir`, `--model-dir`.

## Dashboard

[Streamlit](https://streamlit.io) app (`streamlit_app.py` + `dashboard/`) with interactive
[Plotly](https://plotly.com/python/) charts on a colour-vision-safe palette. Theme and upload limits live in
`.streamlit/config.toml`.

| Page | What it does |
|---|---|
| **[Underwriting](#underwriting)** | Score one applicant, see the decision, the SHAP factor breakdown, plain-language risk reasons and a what-if curve. Downloads a JSON decision report. |
| **[Batch scoring](#batch-scoring)** | Score a CSV of applicants. Score distribution, decision mix, per-row reasons and errors, results download. |
| **[Portfolio insights](#portfolio-insights)** | Observed default rates across 15 applicant segments, plus where risk concentrates. |
| **[Model performance](#model-performance)** | Discrimination, calibration, policy economics, feature importance and fairness, from the test set. |

The dashboard and the API are independent: both load the model bundle in-process through
[`credscore.service.ScoringService`](credscore/service.py), so the dashboard needs no API server running. The model
is loaded once per server process (`st.cache_resource`) and shared by every session.

## REST API

[FastAPI](https://fastapi.tiangolo.com) service in [`backend/app.py`](backend/app.py); interactive docs at `/docs`.

- `GET /api/v1/health` — liveness and loaded model version
- `GET /api/v1/model` — version, test metrics, decision policy, scorecard
- `GET /api/v1/model/performance` — curves, calibration, gains, bands, fairness
- `GET /api/v1/model/importance` — global mean |SHAP| importance
- `GET /api/v1/schema` — applicant fields, allowed category values, defaults
- `GET /api/v1/insights` — portfolio default rates by segment
- `POST /api/v1/score` — score one applicant profile
- `POST /api/v1/score/batch` — score up to 10,000 profiles; bad rows are reported, not fatal
- `POST /api/v1/score/features` — score raw model features (advanced)

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

Applicant fields use human units (years, amounts, counts); the service maps them onto model features. Unspecified
optional fields fall back to the training population's typical value, and the response marks which factors came
from the applicant (`source: "applicant"`) versus a default.

## How it works

```mermaid
flowchart LR
    raw["Kaggle CSVs<br/>dataset/"] --> pipeline["credscore.pipeline<br/>features · train · evaluate · insights"]
    pipeline --> bundle[("models/<br/>model.ubj · metadata<br/>report · insights")]
    bundle --> dashboard["Streamlit dashboard<br/>streamlit_app.py"]
    bundle --> api["FastAPI service<br/>backend/app.py"]
```

### Modelling decisions

- **Split before anything else.** Train/validation/test (70/15/15, stratified) is decided up front. Every reported
  number comes from the test set; the validation set only drives early stopping and the policy cut-offs.
- **No oversampling.** The 8% default rate is left alone. SMOTE on this data interpolates one-hot flags and
  median-imputed columns into values that cannot occur (a fractional `FLAG_PHONE`), so a model learns to detect
  synthetic rows instead of risk, and the resulting probabilities are no longer calibrated — which the credit score
  and the cut-offs depend on.
- **Missing stays missing.** [XGBoost](https://xgboost.readthedocs.io/en/stable/) learns a default direction per
  split, so no imputer is needed and "no credit bureau history" keeps its meaning instead of becoming a median.
- **Native categorical splits.** Category vocabularies are stored in the model bundle, so no encoder has to be
  replicated at serving time and unseen categories degrade to missing.
- **Explanations from the model itself.** Exact TreeSHAP via XGBoost's `pred_contribs`, so there is no separate
  explainer artifact to drift or unpickle.
- **Protected attributes.** `CODE_GENDER` is excluded from the model and retained only for fairness reporting.
  Application-process artifacts (weekday/hour of application) and near-constant flags are dropped too.
- **One feature module.** Training and serving share [`credscore/features.py`](credscore/features.py); derived
  ratios are always recomputed and never accepted as input, so the two cannot drift.
- **Native model format.** `model.ubj` rather than a pickle: portable across versions and safe to load.

Scoring uses a standard scorecard scaling: 650 points = 20:1 good:bad odds, and every 40 points doubles the odds,
clipped to 300–850.

## Project layout

```
credscore/             core package (shared by pipeline, dashboard and API)
  features.py          feature engineering + FeatureSpec (vocabularies, serving defaults)
  profile.py           human-friendly applicant profile -> model features
  model.py             model bundle: booster + spec + metadata
  service.py           scoring, explanations, decisions
  scoring.py           PD -> credit score, risk bands, decision policy
  evaluation.py        metrics, curves, gains, fairness
  labels.py            human-readable feature names and value formatting
  pipeline/            download -> features -> train -> insights (python -m credscore.pipeline)
backend/               FastAPI service (app factory, schemas)
streamlit_app.py       dashboard entrypoint: navigation, model status
frontend/
  dashboard.py         Streamlit Cloud entrypoint (the deployed URL uses it); runs streamlit_app.py
dashboard/             Streamlit pages
  common.py            model loading, formatting, chart styling, example applicants
  underwriting.py      · batch.py · portfolio.py · performance.py
models/                trained bundle: model.ubj, metadata, report, insights
docs/screenshots/      the screenshots in this README
tests/                 pytest suite incl. synthetic-data pipeline fixture
```

## Deployment

- **[Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)** — the live app runs
  `frontend/dashboard.py` from the `main` branch on Python 3.10+, installing `requirements.txt`. The trained bundle
  in `models/` is committed, so no training happens on the server.
- **Docker** — `docker compose up --build` serves the dashboard on 8501 and the API on 8000 from one image, with
  `./models` mounted read-only.
- **GitHub Codespaces** — the dev container in `.devcontainer/` installs the requirements and starts the dashboard.

## Tests

```bash
python -m pytest            # 51 tests, ~20 s
```

The suite builds a synthetic Home Credit-shaped dataset, runs the real pipeline CLI over it, then exercises the API
and every dashboard page (headlessly, via Streamlit's `AppTest`) against that model — so it needs neither the
1.5 GB dataset nor a trained model.

## Configuration

Environment variables, all optional:

- `CREDSCORE_DATA_DIR` — raw CSVs and `features.parquet` (default `dataset/`)
- `CREDSCORE_MODEL_DIR` — model bundle, report and insights (default `models/`)
- `CREDSCORE_APPROVE_PD`, `CREDSCORE_DECLINE_PD` — override the decision cut-offs at serving time
  (default: the cut-offs from training)
- `CREDSCORE_CORS_ORIGINS` — comma-separated CORS allow-list for the API (default `*`)

## Limitations

- Trained on Home Credit's 2018 competition sample; it is a portfolio demonstrator, not a production credit policy.
- `bureau_balance.csv`, `POS_CASH_balance.csv` and `credit_card_balance.csv` are not used; adding them is the most
  obvious route past AUC 0.784.
- Decision cut-offs are set by target approval rates, not by expected loss or profit — real policy needs pricing,
  exposure and regulatory input.
- Fairness reporting covers gender and age bands on the test set only; it is monitoring, not a compliance review.

## Acknowledgements

Data from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition.
Built with [XGBoost](https://xgboost.readthedocs.io/en/stable/), [Streamlit](https://streamlit.io),
[FastAPI](https://fastapi.tiangolo.com), [Plotly](https://plotly.com/python/), [pandas](https://pandas.pydata.org)
and [scikit-learn](https://scikit-learn.org).
