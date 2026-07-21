from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
BINARY_MAP = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}

NOMINAL_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# The exact category set seen at training time for each nominal column, so a
# single-customer request always produces the same one-hot columns that
# pd.get_dummies produced over the full training set (see encode_customer).
NOMINAL_CATEGORIES = {
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

DEFAULT_THRESHOLD = 0.5

app = FastAPI(title="Telco Customer Churn API")

model = joblib.load(MODELS_DIR / "churn_model.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")
shap_background = joblib.load(MODELS_DIR / "shap_background.joblib")

# LinearExplainer computes exact Shapley values for a linear model in closed
# form against this background sample of scaled training rows (see
# src/train_final_model.py) - no per-request retraining or sampling needed.
explainer = shap.LinearExplainer(model, shap_background)


class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72, description="Months, matches the training data range (0-72)")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(
        gt=0, le=150, description="Matches the training data range (max observed was ~$118.75)"
    )


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold_used: float


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class ExplanationResponse(BaseModel):
    churn_probability: float
    base_value: float
    contributions: list[FeatureContribution]


def encode_customer(customer: CustomerInput) -> pd.DataFrame:
    row = pd.DataFrame([customer.model_dump()])

    for col in BINARY_COLS:
        row[col] = row[col].map(BINARY_MAP)

    # fix each nominal column's categories to the full training-time set so
    # get_dummies always yields the same columns, regardless of which single
    # value this one customer happens to have
    for col, categories in NOMINAL_CATEGORIES.items():
        row[col] = pd.Categorical(row[col], categories=categories)

    row = pd.get_dummies(row, columns=NOMINAL_COLS, drop_first=True)

    # add any one-hot columns missing for this single customer, in the training column order
    row = row.reindex(columns=feature_columns, fill_value=0)
    return row


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput, threshold: float = DEFAULT_THRESHOLD) -> PredictionResponse:
    row = encode_customer(customer)
    row_scaled = scaler.transform(row)
    probability = float(model.predict_proba(row_scaled)[0, 1])

    return PredictionResponse(
        churn_probability=probability,
        churn_prediction=probability >= threshold,
        threshold_used=threshold,
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(customer: CustomerInput, top_n: int | None = None) -> ExplanationResponse:
    row = encode_customer(customer)
    row_scaled = scaler.transform(row)
    probability = float(model.predict_proba(row_scaled)[0, 1])

    shap_values = explainer.shap_values(row_scaled)[0]
    contributions = sorted(
        zip(feature_columns, shap_values), key=lambda pair: abs(pair[1]), reverse=True
    )
    if top_n is not None:
        contributions = contributions[:top_n]

    return ExplanationResponse(
        churn_probability=probability,
        base_value=float(explainer.expected_value),
        contributions=[
            FeatureContribution(feature=name, shap_value=float(value)) for name, value in contributions
        ],
    )
