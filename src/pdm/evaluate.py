"""Metrics, threshold selection and report figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_threshold(y_true, scores, beta: float) -> float:
    """Return the probability threshold that maximises the F-beta score."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall pair has no threshold attached.
    precision, recall = precision[:-1], recall[:-1]
    b2 = beta**2
    denom = b2 * precision + recall
    fbeta = np.divide(
        (1 + b2) * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0
    )
    return float(thresholds[int(np.argmax(fbeta))])


def classification_metrics(y_true, scores, threshold: float, beta: float) -> dict:
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(roc_auc_score(y_true, scores), 4),
        "pr_auc": round(average_precision_score(y_true, scores), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(fbeta_score(y_true, y_pred, beta=1, zero_division=0), 4),
        f"f{beta:g}": round(fbeta_score(y_true, y_pred, beta=beta, zero_division=0), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No failure", "Failure"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curves(y_true, curves: dict, path: Path) -> None:
    """curves maps a label to the scores of one model on the same test set."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, scores in curves.items():
        # scikit-learn appends the AP value to the label itself.
        PrecisionRecallDisplay.from_predictions(y_true, scores, name=label, ax=ax)
    base_rate = float(np.mean(y_true))
    random_label = f"Random guess (AP = {base_rate:.2f})"
    ax.axhline(base_rate, linestyle="--", color="grey", label=random_label)
    ax.set_xlabel("Recall (share of real failures caught)")
    ax.set_ylabel("Precision (share of alerts that are real)")
    ax.set_title("Precision-recall on the held-out test set")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(names, means, stds, path: Path) -> None:
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(np.array(names)[order], np.array(means)[order], xerr=np.array(stds)[order])
    ax.set_xlabel("Drop in average precision when the feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
