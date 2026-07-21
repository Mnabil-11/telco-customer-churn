# Telco Customer Churn Prediction

[![CI](https://github.com/Mnabil-11/telco-customer-churn/actions/workflows/ci.yml/badge.svg)](https://github.com/Mnabil-11/telco-customer-churn/actions/workflows/ci.yml)

A learning project: predict which telecom customers are likely to churn,
built on top of SQL (not a plain CSV/DataFrame workflow) and covering the
full path from raw data to a served model, including a business
cost-benefit analysis.

See [MODEL_CARD.md](MODEL_CARD.md) for the technical writeup of model
selection, decision-threshold rationale, class-imbalance handling, and
known limitations.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(Kaggle) — 7,043 customers, 21 columns. Loaded into a SQLite database
(`data/churn.db`) instead of being used directly as a CSV, to practice SQL
exploration.

## Project Structure

```
data/               Raw CSV + SQLite database (not committed)
notebooks/figures/  Saved plots from EDA, evaluation, and business analysis
models/             Persisted final model artifacts (joblib)
src/                Pipeline scripts, run in order (see below)
app/                FastAPI serving app
```

## Pipeline (run in order)

```bash
python -m venv venv
./venv/Scripts/pip install -r requirements.txt   # Windows
# source venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

python src/csv_to_sqlite.py        # CSV -> SQLite table `customers`
python src/clean_data.py           # cleans TotalCharges -> `customers_clean`
python src/eda_visual.py           # numeric distributions + boxplots by churn
python src/eda_categorical.py      # churn rate by categorical variable
python src/feature_engineering.py  # encoding -> `features` table
python src/train_models.py         # trains & compares 3 classifiers
python src/evaluate_models.py      # confusion matrices, precision/recall/F1
python src/business_framing.py     # cost-benefit, threshold tuning
python src/sensitivity_analysis.py # stress-tests the business assumptions
python src/train_final_model.py    # retrains on 100% of data, saves to models/
python src/shap_analysis.py        # SHAP global importance + example explanations
```

## Key Findings

- Baseline churn rate: **26.54%**
- Strongest churn drivers: `Contract` (Month-to-month 42.7% vs Two year
  2.8%), `InternetService = Fiber optic` (41.9%), low `tenure`, missing
  `TechSupport`/`OnlineSecurity`, `PaymentMethod = Electronic check`
- Best model: **Logistic Regression** (79.77% test accuracy, best
  generalization — Random Forest and XGBoost overfit noticeably more)
- At the default 0.5 decision threshold, Recall on churners is only 55%
  (misses ~45% of actual churners) — **this is exactly why 0.5 isn't used
  as the actual operating threshold**: a missed churner is a customer we
  never even tried to retain, and per the business framing below that's
  far costlier than an unnecessary retention offer
- Business framing: given retention-offer economics (offer cost ≈ 20% of
  monthly charges, 30% success rate, 12 months of retained revenue if
  saved), the profit-maximizing decision threshold is much lower than 0.5
  — broad targeting is economically optimal, and this holds even under
  pessimistic cost/success assumptions (sensitivity analysis). At that
  recommended threshold (~0.05), Recall jumps to 98.4% (catching 368 of
  374 actual churners) at the cost of more false alarms — see
  [MODEL_CARD.md](MODEL_CARD.md)'s Evaluation section for the full
  metrics table

## Running the API

```bash
./venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0
  }'
```

Response:

```json
{"churn_probability": 0.65, "churn_prediction": true, "threshold_used": 0.5}
```

Pass `?threshold=0.05` to use the business-optimal cutoff from the
sensitivity analysis instead of the default 0.5.

### Explaining a prediction (SHAP)

`POST /explain` takes the same payload and returns *why* — every
feature's SHAP contribution plus the model's base value, sorted by
magnitude:

```bash
curl -X POST "http://127.0.0.1:8000/explain?top_n=5" \
  -H "Content-Type: application/json" \
  -d '{ ...same fields as /predict... }'
```

```json
{
  "churn_probability": 0.6489,
  "base_value": -1.5256,
  "contributions": [
    {"feature": "tenure", "shap_value": 0.997},
    {"feature": "InternetService_Fiber optic", "shap_value": 0.632},
    {"feature": "Contract_Two year", "shap_value": 0.283},
    {"feature": "MonthlyCharges", "shap_value": -0.268},
    {"feature": "PaymentMethod_Electronic check", "shap_value": 0.196}
  ]
}
```

`base_value + sum(contributions) == logit(churn_probability)`, always —
see [MODEL_CARD.md](MODEL_CARD.md)'s "Model Interpretability (SHAP)"
section for the full global/local analysis and figures.

## Running the API with Docker

The image only needs `app/`, `models/`, and `requirements-api.txt` (a
lighter dependency set than the full pipeline's `requirements.txt`).

```bash
docker build -t telco-churn-api .
docker run -d -p 8000:8000 --name telco-churn-api telco-churn-api
```

Same `/health`, `/predict`, and `/explain` endpoints as above, now served
from the container on port 8000.

### Or with Docker Compose

```bash
docker compose up -d --build
```

Equivalent, but also adds a healthcheck (`docker compose ps` shows
`healthy`) and `restart: unless-stopped`. See `docker-compose.yml`.

## Running Tests

```bash
./venv/Scripts/pip install -r requirements-dev.txt
./venv/Scripts/python.exe -m pytest tests/ -v
```

`tests/test_api.py` covers `/health`, `/predict` and `/explain` (happy
path, threshold behavior, all input validation bounds), and
`encode_customer` in isolation. `tests/test_pipeline.py` runs the entire
`src/*.py` pipeline end-to-end against a small synthetic dataset.

## CI

`.github/workflows/ci.yml` runs the full test suite and a Docker
build+smoke-test on every push/PR to `master`.
