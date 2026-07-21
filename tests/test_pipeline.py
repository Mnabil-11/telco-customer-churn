"""
Integration test for the full analysis pipeline (src/*.py), run end-to-end
against a small synthetic dataset instead of the real 7,043-row CSV.

Each pipeline script keeps its I/O paths as module-level constants
(DB_PATH, FIG_DIR, MODELS_DIR, CSV_PATH) computed at import time, so this
test monkeypatches those constants on the already-imported modules to point
into a temporary sandbox, then calls each script's main() in the same order
a real run would use. This catches wiring bugs between stages (a stage
expecting a table/column another stage didn't produce) without touching
any script's source structure.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

import src.business_framing as business_framing
import src.clean_data as clean_data
import src.csv_to_sqlite as csv_to_sqlite
import src.eda_categorical as eda_categorical
import src.eda_visual as eda_visual
import src.evaluate_models as evaluate_models
import src.feature_engineering as feature_engineering
import src.sensitivity_analysis as sensitivity_analysis
import src.train_final_model as train_final_model
import src.train_models as train_models


def make_synthetic_customers(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, n)
    tenure[:2] = 0  # guarantee the brand-new-customer edge case Stage 3 handles
    monthly_charges = np.round(rng.uniform(18.0, 118.0, n), 2)

    total_charges = [
        "" if t == 0 else str(round(t * m + rng.uniform(-5, 5), 2))
        for t, m in zip(tenure, monthly_charges)
    ]

    return pd.DataFrame(
        {
            "customerID": [f"synth-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.85, 0.15]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
            ),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        }
    )


@pytest.fixture
def pipeline_paths(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    fig_dir = tmp_path / "notebooks" / "figures"
    models_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True)

    csv_path = data_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    db_path = data_dir / "churn.db"

    synthetic = make_synthetic_customers()
    synthetic.to_csv(csv_path, index=False)

    for module in (
        clean_data, eda_visual, eda_categorical, feature_engineering,
        train_models, evaluate_models, business_framing, sensitivity_analysis,
        train_final_model,
    ):
        monkeypatch.setattr(module, "DB_PATH", db_path)
    monkeypatch.setattr(csv_to_sqlite, "CSV_PATH", csv_path)
    monkeypatch.setattr(csv_to_sqlite, "DB_PATH", db_path)

    for module in (eda_visual, eda_categorical, evaluate_models, business_framing, sensitivity_analysis):
        monkeypatch.setattr(module, "FIG_DIR", fig_dir)
    monkeypatch.setattr(train_final_model, "MODELS_DIR", models_dir)

    return {
        "synthetic": synthetic,
        "csv_path": csv_path,
        "db_path": db_path,
        "fig_dir": fig_dir,
        "models_dir": models_dir,
    }


def test_full_pipeline_runs_end_to_end(pipeline_paths, capsys) -> None:
    synthetic = pipeline_paths["synthetic"]
    db_path = pipeline_paths["db_path"]
    fig_dir = pipeline_paths["fig_dir"]
    models_dir = pipeline_paths["models_dir"]

    # Stage 1: CSV -> SQLite
    csv_to_sqlite.main()
    conn = sqlite3.connect(db_path)
    raw = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    assert len(raw) == len(synthetic)

    # Stage 3: cleaning
    clean_data.main()
    conn = sqlite3.connect(db_path)
    cleaned = pd.read_sql("SELECT * FROM customers_clean", conn)
    conn.close()
    assert cleaned["TotalCharges"].isna().sum() == 0
    # the two forced brand-new customers (tenure=0, blank TotalCharges) must become 0.0
    assert (cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"] == 0.0).all()

    # Stage 3: EDA figures
    eda_visual.main()
    eda_categorical.main()
    assert (fig_dir / "histograms.png").exists()
    assert (fig_dir / "boxplots_by_churn.png").exists()
    assert (fig_dir / "categorical_churn_rates.png").exists()

    # Stage 4: feature engineering
    feature_engineering.main()
    conn = sqlite3.connect(db_path)
    features = pd.read_sql("SELECT * FROM features", conn)
    conn.close()
    assert "customerID" not in features.columns
    assert "TotalCharges" not in features.columns
    assert features.isna().sum().sum() == 0
    assert all(dtype.kind in "biuf" for dtype in features.dtypes)
    assert set(features["Churn"].unique()) <= {0, 1}
    # cleaning must not have changed how many customers churned
    assert features["Churn"].sum() == (synthetic["Churn"] == "Yes").sum()

    # Stage 5: train & compare models (smoke test via captured output)
    train_models.main()
    output = capsys.readouterr().out
    assert "Logistic Regression" in output
    assert "Random Forest" in output
    assert "XGBoost" in output

    # Stage 6: evaluation
    evaluate_models.main()
    assert (fig_dir / "confusion_matrices.png").exists()

    # Stage 7: business framing + sensitivity analysis
    business_framing.main()
    assert (fig_dir / "business_threshold_tuning.png").exists()

    sensitivity_analysis.main()
    assert (fig_dir / "sensitivity_analysis.png").exists()

    # Stage 8: final model persisted and usable
    train_final_model.main()
    assert (models_dir / "churn_model.joblib").exists()
    assert (models_dir / "scaler.joblib").exists()
    assert (models_dir / "feature_columns.joblib").exists()

    import joblib

    model = joblib.load(models_dir / "churn_model.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    feature_columns = joblib.load(models_dir / "feature_columns.joblib")

    X = features[feature_columns]
    probabilities = model.predict_proba(scaler.transform(X))[:, 1]
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
