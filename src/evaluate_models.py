import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE), False),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE), False),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (name, (model, needs_scaling)) in zip(axes, models.items()):
        if needs_scaling:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        print(f"=== {name} ===")
        print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrices to {FIG_DIR / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
