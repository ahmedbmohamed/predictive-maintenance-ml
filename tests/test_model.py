from pdm.features import add_engineered_features
from pdm.predictor import Predictor, explain


def test_training_selects_a_tree_model_and_beats_baseline(trained_bundle):
    report = trained_bundle["report"]
    cv = report["model_comparison_cv"]
    assert report["selected_model"] in {"random_forest", "gradient_boosting"}
    best = cv[report["selected_model"]]["cv_pr_auc_mean"]
    assert best > cv["logistic_regression"]["cv_pr_auc_mean"]


def test_model_quality_floor(trained_bundle):
    m = trained_bundle["report"]["test_metrics"]
    # Random guessing gives a PR-AUC of about 0.034 on this dataset.
    assert m["pr_auc"] > 0.7
    assert m["recall"] > 0.7
    assert m["roc_auc"] > 0.9


def test_physics_features_improve_the_model(trained_bundle):
    report = trained_bundle["report"]
    with_physics = report["test_metrics"]["pr_auc"]
    raw_only = report["ablation_without_engineered_features"]["pr_auc"]
    assert with_physics > raw_only


def test_predictor_output_schema(trained_bundle, normal_reading):
    result = Predictor.load(trained_bundle["model_path"]).predict(normal_reading)[0]
    assert set(result) == {
        "failure_probability",
        "failure_predicted",
        "anomaly_score",
        "is_anomaly",
        "status",
        "reasons",
    }
    assert 0 <= result["failure_probability"] <= 1
    assert result["status"] in {"OK", "ANOMALY", "FAILURE_RISK"}


def test_overstrain_is_riskier_than_normal(trained_bundle, normal_reading, overstrained_reading):
    predictor = Predictor.load(trained_bundle["model_path"])
    normal = predictor.predict(normal_reading)[0]
    risky = predictor.predict(overstrained_reading)[0]
    assert normal["status"] == "OK"
    assert risky["failure_predicted"]
    assert risky["failure_probability"] > normal["failure_probability"]


def test_explanations(normal_reading, overstrained_reading):
    assert explain(add_engineered_features(normal_reading).iloc[0]) == []
    reasons = explain(add_engineered_features(overstrained_reading).iloc[0])
    assert any(r.startswith("Overstrain risk") for r in reasons)
    assert any(r.startswith("Tool near end of life") for r in reasons)
