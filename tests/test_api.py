import pytest
from fastapi.testclient import TestClient

from app.main import app, encode_customer, feature_columns, CustomerInput

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
}


def customer(**overrides: object) -> dict:
    return {**VALID_CUSTOMER, **overrides}


# --- /health ---


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- /predict: happy path ---


def test_predict_valid_customer_returns_expected_shape() -> None:
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"churn_probability", "churn_prediction", "threshold_used"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["threshold_used"] == 0.5


def test_predict_high_risk_vs_low_risk_customer_ordering() -> None:
    high_risk = customer(
        tenure=1, Contract="Month-to-month", InternetService="Fiber optic",
        TechSupport="No", OnlineSecurity="No", PaymentMethod="Electronic check",
    )
    low_risk = customer(
        tenure=60, Contract="Two year", InternetService="DSL",
        TechSupport="Yes", OnlineSecurity="Yes", PaymentMethod="Bank transfer (automatic)",
    )

    high_prob = client.post("/predict", json=high_risk).json()["churn_probability"]
    low_prob = client.post("/predict", json=low_risk).json()["churn_probability"]

    assert high_prob > low_prob


def test_predict_custom_threshold_changes_prediction() -> None:
    borderline = customer(tenure=1, MonthlyCharges=90.0)
    probability = client.post("/predict", json=borderline).json()["churn_probability"]

    low_threshold_response = client.post(
        "/predict", json=borderline, params={"threshold": max(probability - 0.05, 0.001)}
    )
    high_threshold_response = client.post(
        "/predict", json=borderline, params={"threshold": min(probability + 0.05, 0.999)}
    )

    assert low_threshold_response.json()["churn_prediction"] is True
    assert high_threshold_response.json()["churn_prediction"] is False


# --- /predict: validation ---


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenure": -5},
        {"tenure": 500},
        {"MonthlyCharges": 0},
        {"MonthlyCharges": -10},
        {"MonthlyCharges": 200},
        {"Contract": "Three year"},
        {"SeniorCitizen": 2},
        {"InternetService": "Cable"},
    ],
)
def test_predict_rejects_out_of_range_or_invalid_values(overrides: dict) -> None:
    response = client.post("/predict", json=customer(**overrides))
    assert response.status_code == 422


def test_predict_rejects_wrong_type() -> None:
    response = client.post("/predict", json=customer(tenure="abc"))
    assert response.status_code == 422


def test_predict_coerces_numeric_string() -> None:
    response = client.post("/predict", json=customer(MonthlyCharges="85.0"))
    assert response.status_code == 200


def test_predict_rejects_missing_required_field() -> None:
    payload = VALID_CUSTOMER.copy()
    del payload["MonthlyCharges"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# --- encode_customer: unit-level ---


def test_encode_customer_produces_training_column_order() -> None:
    row = encode_customer(CustomerInput(**VALID_CUSTOMER))
    assert list(row.columns) == feature_columns


def test_encode_customer_maps_binary_fields_to_0_1() -> None:
    row = encode_customer(CustomerInput(**customer(gender="Male", Partner="Yes")))
    assert row["gender"].iloc[0] == 1
    assert row["Partner"].iloc[0] == 1


def test_encode_customer_one_hot_encodes_contract() -> None:
    two_year = encode_customer(CustomerInput(**customer(Contract="Two year")))
    month_to_month = encode_customer(CustomerInput(**customer(Contract="Month-to-month")))

    assert two_year["Contract_Two year"].iloc[0] == 1
    assert two_year["Contract_One year"].iloc[0] == 0
    # Month-to-month is the dropped baseline category: all Contract_* columns are 0
    assert month_to_month["Contract_Two year"].iloc[0] == 0
    assert month_to_month["Contract_One year"].iloc[0] == 0
