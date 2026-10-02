"""Train the fraud model with an honest evaluation setup.

Usage:
    python train_model.py --data "path/to/AIML Dataset.csv"

Split: 60% train / 20% validation / 20% test (stratified).
- Models are fit on train.
- The decision threshold is tuned on validation only.
- Final metrics are reported once on the untouched test set.
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from features import build_features

SEED = 42


def best_f1_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[f1.argmax()])


def main(data_path: str) -> None:
    raw = pd.read_csv(data_path)
    y = raw["isFraud"]
    X = build_features(raw)  # isFlaggedFraud and account names are not used

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        thr = best_f1_threshold(y_val, val_proba)  # tuned on validation only
        pred = (test_proba >= thr).astype(int)
        rows.append({
            "Model": name,
            "Threshold": round(thr, 3),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "PR-AUC": average_precision_score(y_test, test_proba),
            "ROC-AUC": roc_auc_score(y_test, test_proba),
        })
    results = pd.DataFrame(rows).round(4)
    print(results.to_string(index=False))

    # Pick the final model by validation PR-AUC (not by test results).
    best_name = max(
        fitted, key=lambda n: average_precision_score(y_val, fitted[n].predict_proba(X_val)[:, 1])
    )
    best = fitted[best_name]
    threshold = best_f1_threshold(y_val, best.predict_proba(X_val)[:, 1])
    test_pred = (best.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    print(f"\nFinal model: {best_name} (threshold {threshold:.3f}, chosen on validation)")
    print("Confusion matrix on test [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, test_pred))

    importances = (
        pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
        if hasattr(best, "feature_importances_") else None
    )
    if importances is not None:
        print("\nTop features:\n", importances.head(8).round(4).to_string())

    joblib.dump(best, "fraud_model.pkl", compress=3)
    meta = {
        "model": best_name,
        "threshold": threshold,
        "features": list(X.columns),
        "sklearn_version": sklearn.__version__,
        "test_metrics": results[results["Model"] == best_name].iloc[0].to_dict(),
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    results.to_csv("results.csv", index=False)
    print("\nSaved fraud_model.pkl, model_meta.json, results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the PaySim CSV")
    main(parser.parse_args().data)
