import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "churn.db"
FIG_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PaperlessBilling",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "StreamingTV",
    "MultipleLines",
]


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customers_clean", conn)
    conn.close()

    df["churn_flag"] = (df["Churn"] == "Yes").astype(int)

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, CATEGORICAL_COLS):
        churn_rate = df.groupby(col)["churn_flag"].mean().sort_values(ascending=False) * 100
        churn_rate.plot(kind="bar", ax=ax)
        ax.set_title(col)
        ax.set_ylabel("Churn rate (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "categorical_churn_rates.png", dpi=120)
    plt.close(fig)

    print(f"Saved figure to {FIG_DIR / 'categorical_churn_rates.png'}")


if __name__ == "__main__":
    main()
