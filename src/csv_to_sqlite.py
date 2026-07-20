import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_PATH = DATA_DIR / "churn.db"


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Loaded {len(df)} rows into {DB_PATH} (table: customers)")


if __name__ == "__main__":
    main()
