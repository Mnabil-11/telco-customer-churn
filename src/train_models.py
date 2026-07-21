import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "churn.db"
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

    print(f"Train: {len(X_train)} rows ({y_train.mean()*100:.2f}% churn)")
    print(f"Test:  {len(X_test)} rows ({y_test.mean()*100:.2f}% churn)")
    print()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE), False),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE), False),
    }

    for name, (model, needs_scaling) in models.items():
        if needs_scaling:
            model.fit(X_train_scaled, y_train)
            train_acc = model.score(X_train_scaled, y_train)
            test_acc = model.score(X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)

        print(f"{name}: train accuracy = {train_acc:.4f}, test accuracy = {test_acc:.4f}")


if __name__ == "__main__":
    main()
