"""
Évaluation et comparaison des modèles.
Métriques : Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
Analyse d'erreurs : matrice de confusion, courbe ROC, courbe PR.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance


def _find_best_threshold(y_true, y_proba) -> float:
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = [f1_score(y_true, y_proba >= t, zero_division=0) for t in thresholds]
    return float(thresholds[np.argmax(f1_scores)])


def evaluate_model(model, X_test, y_test, model_name: str = "") -> dict:
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    if y_proba is not None:
        threshold = _find_best_threshold(y_test, y_proba)
        y_pred = (y_proba >= threshold).astype(int)
    else:
        threshold = 0.5
        y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "threshold": round(threshold, 2),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "pr_auc": average_precision_score(y_test, y_proba) if y_proba is not None else None,
    }
    return metrics


def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        row = evaluate_model(model, X_test, y_test, model_name=name)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("f1", ascending=False)


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "", ax=None):
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    if y_proba is not None:
        threshold = _find_best_threshold(y_test, y_proba)
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        threshold = 0.5
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False)
    if ax is not None:
        ax.set_title(f"{model_name}\n(seuil={threshold:.2f})")
    return cm


def plot_roc_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Courbes ROC — Comparaison des modèles")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pr_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", label=f"Baseline (random) = {baseline:.2f}")
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Courbes Precision-Recall — Comparaison des modèles")
    ax.legend()
    fig.tight_layout()
    return fig


def get_feature_importance(model, feature_names: list[str]) -> pd.Series:
    """Récupère l'importance native pour les modèles basés sur les arbres."""
    clf = model.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    raise AttributeError("Ce modèle ne supporte pas feature_importances_")


def get_permutation_importance(model, X_test, y_test, feature_names: list[str], n_repeats: int = 10) -> pd.Series:
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="f1",
    )
    return pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
