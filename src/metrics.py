"""Métricas de classificação binária."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = (probs >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        out["auc"] = float("nan")
    return out
