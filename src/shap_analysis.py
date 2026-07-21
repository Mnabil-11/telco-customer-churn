import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "churn.db"
FIG_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"
RANDOM_STATE = 42


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

    # LinearExplainer computes exact Shapley values for linear models in
    # closed form (no sampling/approximation needed, unlike tree/kernel
    # explainers), using X_train_scaled as the background distribution.
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)
    shap_values.feature_names = list(X.columns)

    # Global: mean |SHAP value| per feature across the whole test set
    fig = plt.figure(figsize=(9, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "shap_global_importance.png", dpi=120)
    plt.close(fig)

    # Global: beeswarm shows both importance AND direction (does a high
    # value of this feature push toward or away from churn?)
    fig = plt.figure(figsize=(9, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "shap_beeswarm.png", dpi=120)
    plt.close(fig)

    # Local: explain the lowest-risk and highest-risk test customers end to end
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    lowest_risk_idx = int(np.argmin(probabilities))
    highest_risk_idx = int(np.argmax(probabilities))

    for idx, name in [(lowest_risk_idx, "lowest_risk"), (highest_risk_idx, "highest_risk")]:
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"shap_waterfall_{name}.png", dpi=120)
        plt.close(fig)

    print(f"Lowest-risk test customer: probability={probabilities[lowest_risk_idx]:.4f}")
    print(f"Highest-risk test customer: probability={probabilities[highest_risk_idx]:.4f}")

    print("Top 10 features by mean |SHAP value| across the test set:")
    importance = pd.Series(
        abs(shap_values.values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)
    print(importance.head(10))
    print(f"\nSaved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
