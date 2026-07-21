import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "churn.db"

BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
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


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customers_clean", conn)

    df = df.drop(columns=["customerID", "TotalCharges"])

    for col in BINARY_COLS:
        df[col] = df[col].map(BINARY_MAP)

    df = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=True)

    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()

    print(f"features table: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    print("Columns:")
    for col in df.columns:
        print(" -", col)


if __name__ == "__main__":
    main()
