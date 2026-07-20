import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "churn.db"
FIG_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customers_clean", conn)
    conn.close()

    # Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.histplot(df[col], bins=30, ax=ax)
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "histograms.png", dpi=120)
    plt.close(fig)

    # Boxplots split by Churn
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.boxplot(data=df, x="Churn", y=col, ax=ax)
        ax.set_title(f"{col} by Churn")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boxplots_by_churn.png", dpi=120)
    plt.close(fig)

    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
