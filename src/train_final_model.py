import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "churn.db"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42


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

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "churn_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(list(X.columns), MODELS_DIR / "feature_columns.joblib")

    print(f"Trained on {len(X)} rows, {X.shape[1]} features")
    print(f"Saved model artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
