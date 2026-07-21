import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "churn.db"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42
SHAP_BACKGROUND_SIZE = 100


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM features", conn)
    conn.close()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_scaled, y)

    # a small random sample of scaled training rows, used as the SHAP
    # LinearExplainer's background distribution at serving time (see
    # app/main.py's /explain endpoint) instead of shipping the full
    # training set to the API
    rng = np.random.default_rng(RANDOM_STATE)
    background_size = min(SHAP_BACKGROUND_SIZE, len(X_scaled))
    sample_idx = rng.choice(len(X_scaled), size=background_size, replace=False)
    shap_background = X_scaled[sample_idx]

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "churn_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(list(X.columns), MODELS_DIR / "feature_columns.joblib")
    joblib.dump(shap_background, MODELS_DIR / "shap_background.joblib")

    print(f"Trained on {len(X)} rows, {X.shape[1]} features")
    print(f"Saved model artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
