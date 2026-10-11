"""Train, compare and evaluate the failure prediction models.

Run with:  python -m pdm.train

Steps
1. Stratified 80/20 split. The test set is used once, at the very end.
2. Three candidate models compared with 5-fold cross-validation on the
   training set, ranked by PR-AUC (the right metric for a 3.4% positive rate).
3. Decision threshold of the best model tuned on out-of-fold predictions
   to maximise F2 (recall matters more than precision in maintenance).
4. Ablation: the same model without the physics-based features.
5. Isolation Forest trained on healthy machines only, as an unsupervised
   detector for abnormal behaviour that was never labelled.
6. Final evaluation on the test set, figures, metrics.json and model bundle.
"""

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pdm import __version__
from pdm.config import (
    CV_FOLDS,
    F_BETA,
    FAILURE_MODES,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SENSOR_COLUMNS,
    TARGET,
    TEST_SIZE,
)
from pdm.data import label_consistency_report, load_dataset
from pdm.evaluate import (
    best_threshold,
    classification_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
)
from pdm.features import FEATURE_COLUMNS, TYPE_COLUMNS, build_features

# Healthy machines flagged by the Isolation Forest are capped at this rate.
ANOMALY_FALSE_ALARM_RATE = 0.03


def candidate_models(n_estimators: int = 400) -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def compare_models(models: dict, X, y, cv) -> dict:
    results = {}
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv, scoring=["average_precision", "roc_auc"], n_jobs=1
        )
        results[name] = {
            "cv_pr_auc_mean": round(float(scores["test_average_precision"].mean()), 4),
            "cv_pr_auc_std": round(float(scores["test_average_precision"].std()), 4),
            "cv_roc_auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
        }
        print(
            f"  {name:22s} PR-AUC {results[name]['cv_pr_auc_mean']:.3f} "
            f"(+/- {results[name]['cv_pr_auc_std']:.3f})"
        )
    return results


def fit_with_threshold(model, X, y, cv):
    """Tune the threshold on out-of-fold scores, then refit on all of X."""
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    threshold = best_threshold(y, oof, beta=F_BETA)
    fitted = clone(model).fit(X, y)
    return fitted, threshold


def train_isolation_forest(X_healthy: pd.DataFrame):
    detector = make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=300, contamination="auto", random_state=RANDOM_STATE),
    )
    detector.fit(X_healthy)
    # score_samples is high for normal points, so it is negated to get an
    # anomaly score where higher means more abnormal.
    healthy_scores = -detector.score_samples(X_healthy)
    threshold = float(np.quantile(healthy_scores, 1 - ANOMALY_FALSE_ALARM_RATE))
    return detector, threshold


def recall_by_failure_mode(test_df: pd.DataFrame, y_pred) -> dict:
    """Share of each failure type the model catches (analysis only)."""
    out = {}
    for mode in FAILURE_MODES:
        mask = (test_df[mode] == 1).to_numpy()
        if mask.sum():
            out[mode] = {
                "count": int(mask.sum()),
                "recall": round(float(np.asarray(y_pred)[mask].mean()), 3),
            }
    return out


def main(n_estimators: int = 400, cv_folds: int = CV_FOLDS, make_figures: bool = True) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and validating data")
    df = load_dataset()
    X = build_features(df)
    y = df[TARGET].to_numpy()

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_df = df.iloc[test_idx]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    print(f"Comparing models with {cv_folds}-fold cross-validation")
    models = candidate_models(n_estimators)
    comparison = compare_models(models, X_train, y_train, cv)
    best_name = max(comparison, key=lambda k: comparison[k]["cv_pr_auc_mean"])
    print(f"Selected model: {best_name}")

    model, threshold = fit_with_threshold(models[best_name], X_train, y_train, cv)
    test_scores = model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= threshold).astype(int)
    final_metrics = classification_metrics(y_test, test_scores, threshold, F_BETA)
    default_metrics = classification_metrics(y_test, test_scores, 0.5, F_BETA)

    print("Ablation without physics-based features")
    raw_columns = SENSOR_COLUMNS + TYPE_COLUMNS
    raw_model, raw_threshold = fit_with_threshold(
        models[best_name], X_train[raw_columns], y_train, cv
    )
    raw_scores = raw_model.predict_proba(X_test[raw_columns])[:, 1]
    ablation = classification_metrics(y_test, raw_scores, raw_threshold, F_BETA)

    print("Training Isolation Forest on healthy machines")
    detector, anomaly_threshold = train_isolation_forest(X_train[y_train == 0])
    anomaly_scores = -detector.score_samples(X_test)
    anomaly_metrics = classification_metrics(y_test, anomaly_scores, anomaly_threshold, F_BETA)

    print("Permutation importance")
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_table = {
        name: round(float(m), 4)
        for name, m in sorted(
            zip(FEATURE_COLUMNS, importance.importances_mean, strict=True), key=lambda t: -t[1]
        )
    }

    report = {
        "model_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "rows": int(len(df)),
            "failures": int(y.sum()),
            "failure_rate": round(float(y.mean()), 4),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "label_consistency": label_consistency_report(df),
        },
        "model_comparison_cv": comparison,
        "selected_model": best_name,
        "test_metrics": final_metrics,
        "test_metrics_default_threshold": default_metrics,
        "ablation_without_engineered_features": ablation,
        "isolation_forest_test_metrics": anomaly_metrics,
        "recall_by_failure_mode": recall_by_failure_mode(test_df, test_pred),
        "permutation_importance": importance_table,
    }

    bundle = {
        "model": model,
        "model_name": best_name,
        "threshold": threshold,
        "anomaly_detector": detector,
        "anomaly_threshold": anomaly_threshold,
        "feature_columns": FEATURE_COLUMNS,
        "version": __version__,
        "trained_at": report["trained_at"],
        "test_metrics": final_metrics,
    }
    joblib.dump(bundle, MODELS_DIR / "model.joblib", compress=3)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    if make_figures:
        plot_confusion_matrix(
            y_test,
            test_pred,
            FIGURES_DIR / "confusion_matrix.png",
            f"{best_name} (threshold {threshold:.2f})",
        )
        plot_pr_curves(
            y_test,
            {
                f"{best_name} + physics features": test_scores,
                f"{best_name}, raw sensors only": raw_scores,
                "Isolation Forest (unsupervised)": anomaly_scores,
            },
            FIGURES_DIR / "precision_recall.png",
        )
        plot_feature_importance(
            FEATURE_COLUMNS,
            importance.importances_mean,
            importance.importances_std,
            FIGURES_DIR / "feature_importance.png",
        )

    print(json.dumps(final_metrics, indent=2))
    return report


if __name__ == "__main__":
    main()
