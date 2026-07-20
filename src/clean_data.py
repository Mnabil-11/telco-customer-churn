import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "churn.db"


def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("SELECT * FROM customers", conn)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df.to_sql("customers_clean", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Wrote {len(df)} rows into customers_clean")


if __name__ == "__main__":
    main()
