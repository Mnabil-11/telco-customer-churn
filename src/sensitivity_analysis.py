import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from business_framing import net_benefit

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "churn.db"
FIG_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"
RANDOM_STATE = 42

SCENARIOS = {
    "Base case (20% cost, 30% success)": dict(offer_cost_rate=0.20, retention_months=12, success_rate=0.30),
    "Higher offer cost (50%)": dict(offer_cost_rate=0.50, retention_months=12, success_rate=0.30),
    "Lower success rate (15%)": dict(offer_cost_rate=0.20, retention_months=12, success_rate=0.15),
    "Pessimistic (50% cost + 15% success)": dict(offer_cost_rate=0.50, retention_months=12, success_rate=0.15),
}


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
    thresholds = np.arange(0.05, 0.96, 0.05)
    n_test = len(y_test)
    scale_factor = 7043 / n_test

    fig, ax = plt.subplots(figsize=(9, 6))

    print(f"{'Scenario':<38}{'Breakeven prob.':<18}{'Best threshold':<18}{'Best benefit (all customers)'}")
    print("-" * 100)

    for label, params in SCENARIOS.items():
        breakeven = params["offer_cost_rate"] / (params["success_rate"] * params["retention_months"])

        benefits = []
        for t in thresholds:
            y_pred_t = (proba >= t).astype(int)
            benefits.append(net_benefit(y_test, y_pred_t, monthly_charges, **params))

        best_idx = int(np.argmax(benefits))
        best_threshold = thresholds[best_idx]
        best_benefit = benefits[best_idx] * scale_factor

        print(f"{label:<38}{breakeven*100:>6.1f}%{'':<11}{best_threshold:<18.2f}${best_benefit:,.0f}")

        ax.plot(thresholds, [b * scale_factor for b in benefits], marker="o", label=label)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Net benefit, extrapolated to 7043 customers ($)")
    ax.set_title("Net benefit vs threshold across cost/success assumptions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity_analysis.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved plot to {FIG_DIR / 'sensitivity_analysis.png'}")


if __name__ == "__main__":
    main()
