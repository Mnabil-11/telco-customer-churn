import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "churn.db"
FIG_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"
RANDOM_STATE = 42

OFFER_COST_RATE = 0.20   # of MonthlyCharges
RETENTION_MONTHS = 12    # months of MonthlyCharges protected if saved
SUCCESS_RATE = 0.30      # chance a retention offer actually works


def net_benefit(
    y_true: pd.Series,
    y_pred: np.ndarray,
    monthly_charges: pd.Series,
    offer_cost_rate: float = OFFER_COST_RATE,
    retention_months: float = RETENTION_MONTHS,
    success_rate: float = SUCCESS_RATE,
) -> float:
    offer_cost = offer_cost_rate * monthly_charges
    retention_value = retention_months * monthly_charges

    targeted = y_pred == 1
    actual_churn = y_true == 1

    tp = targeted & actual_churn
    fp = targeted & ~actual_churn
    fn = ~targeted & actual_churn

    benefit = (success_rate * retention_value[tp] - offer_cost[tp]).sum()
    benefit -= offer_cost[fp].sum()
    benefit -= (success_rate * retention_value[fn]).sum()  # forfeited opportunity
    return benefit


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM features", conn)
    conn.close()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    monthly_charges = X_test["MonthlyCharges"]

    # default threshold 0.5
    y_pred_default = (proba >= 0.5).astype(int)
    benefit_default = net_benefit(y_test, y_pred_default, monthly_charges)

    # sweep thresholds
    thresholds = np.arange(0.05, 0.96, 0.05)
    benefits = []
    for t in thresholds:
        y_pred_t = (proba >= t).astype(int)
        benefits.append(net_benefit(y_test, y_pred_t, monthly_charges))

    best_idx = int(np.argmax(benefits))
    best_threshold = thresholds[best_idx]
    best_benefit = benefits[best_idx]

    n_test = len(y_test)
    scale_factor = 7043 / n_test  # extrapolate test-set result to full customer base

    print("=== At default threshold 0.5 ===")
    print(f"Net benefit on {n_test} test customers: ${benefit_default:,.2f}")
    print(f"Extrapolated to all 7043 customers: ${benefit_default * scale_factor:,.2f}")
    print()
    print(f"=== Best threshold: {best_threshold:.2f} ===")
    print(f"Net benefit on {n_test} test customers: ${best_benefit:,.2f}")
    print(f"Extrapolated to all 7043 customers: ${best_benefit * scale_factor:,.2f}")
    print()
    print(f"Benefit improvement from threshold tuning: ${(best_benefit - benefit_default) * scale_factor:,.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, benefits, marker="o")
    ax.axvline(0.5, color="gray", linestyle="--", label="Default threshold (0.5)")
    ax.axvline(best_threshold, color="green", linestyle="--", label=f"Best threshold ({best_threshold:.2f})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Net benefit on test set ($)")
    ax.set_title("Net financial benefit vs decision threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "business_threshold_tuning.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved plot to {FIG_DIR / 'business_threshold_tuning.png'}")


if __name__ == "__main__":
    main()
