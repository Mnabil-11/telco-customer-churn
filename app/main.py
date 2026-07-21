from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
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

DEFAULT_THRESHOLD = 0.5

app = FastAPI(title="Telco Customer Churn API")

model = joblib.load(MODELS_DIR / "churn_model.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")


class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
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
    MonthlyCharges: float = Field(gt=0, description="Must be a positive amount")


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold_used: float


def encode_customer(customer: CustomerInput) -> pd.DataFrame:
    row = pd.DataFrame([customer.model_dump()])

    for col in BINARY_COLS:
        row[col] = row[col].map(BINARY_MAP)

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
