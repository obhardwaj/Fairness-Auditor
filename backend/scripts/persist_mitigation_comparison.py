"""
Persists baseline + all three mitigation methods (reweighing, exponentiated_gradient,
threshold_optimizer) as MitigationResult rows for both logistic_regression and
gradient_boosting, on the race-binary COMPAS subset.

Methodological notes (see project write-up for full detail):
- ExponentiatedGradient metrics use Fairlearn's internal _pmf_predict thresholded
  at 0.5 (deterministic) rather than the stochastic .predict() -- this is an
  approximation of, not identical to, the classifier the fairness guarantee
  formally covers.
- ThresholdOptimizer's .predict() uses randomized interpolation internally;
  np.random.seed(42) is set immediately before each call to make results
  reproducible across runs.

Run with: docker compose exec api python -m scripts.persist_mitigation_comparison
"""
import numpy as np
import joblib

from app.core.db import SessionLocal
from app.models.models import AuditRun, AuditStatus, MitigationResult, MLModel
from app.ml.compas_pipeline import load_and_filter_compas, filter_to_binary_race, build_feature_matrix
from app.ml.metrics import run_full_metric_suite
from app.ml.mitigation import apply_reweighing, apply_exponentiated_gradient, apply_threshold_optimizer

CSV_PATH = "data/compas-scores-two-years.csv"
ARTIFACT_DIR = "data/artifacts"


def get_or_create_audit_run(db, model_row) -> AuditRun:
    existing = db.query(AuditRun).filter(AuditRun.model_id == model_row.id).first()
    if existing:
        return existing
    audit_run = AuditRun(model_id=model_row.id, status=AuditStatus.completed)
    db.add(audit_run)
    db.commit()
    db.refresh(audit_run)
    return audit_run


def already_persisted(db, audit_run_id: str, method: str) -> bool:
    """Dedupe check: skip inserting if this (audit_run, method) combo already exists."""
    existing = db.query(MitigationResult).filter(
        MitigationResult.audit_run_id == audit_run_id,
        MitigationResult.method == method,
    ).first()
    return existing is not None


def strip_cis_to_point_estimates(suite: dict, calibration_note: str | None = None) -> dict:
    out = {}
    for metric_name, result in suite.items():
        if metric_name == "calibration_within_groups":
            continue
        point, lower, upper = result
        out[metric_name] = {"value": point, "ci_lower": lower, "ci_upper": upper}

    if calibration_note is not None:
        out["calibration_within_groups"] = {"value": None, "note": calibration_note}

    return out


def main():
    db = SessionLocal()

    df = load_and_filter_compas(CSV_PATH)
    df = filter_to_binary_race(df)
    X, y, protected_attr_cols, protected_attrs_raw = build_feature_matrix(df)
    race_values = protected_attrs_raw["race"].to_numpy()

    # --- Baseline ---
    baseline_artifact_paths = {
        "logistic_regression": f"{ARTIFACT_DIR}/compas_logistic_regression_v1.joblib",
        "gradient_boosting": f"{ARTIFACT_DIR}/compas_gradient_boosting_v1.joblib",
    }
    baseline_results = {}
    for algorithm, path in baseline_artifact_paths.items():
        clf = joblib.load(path)
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]
        suite = run_full_metric_suite(
            y_true=y.to_numpy(), y_pred=y_pred, y_prob=y_prob,
            protected_attr=race_values, n_bootstrap=500,
        )
        acc = float((y_pred == y.to_numpy()).mean())
        baseline_results[algorithm] = {"accuracy": acc, "suite": suite}

    # --- Reweighing (pre) ---
    reweigh_results = apply_reweighing(
        X=X, y=y, protected_attr_col_name="race",
        protected_attr_values=race_values, artifact_dir=ARTIFACT_DIR,
    )
    reweighed_metrics = {}
    for algorithm, r in reweigh_results.items():
        y_test = r["y_test"].to_numpy()
        y_pred = r["model"].predict(r["X_test"])
        y_prob = r["model"].predict_proba(r["X_test"])[:, 1]
        suite = run_full_metric_suite(
            y_true=y_test, y_pred=y_pred, y_prob=y_prob,
            protected_attr=r["test_protected_attr"], n_bootstrap=500,
        )
        reweighed_metrics[algorithm] = {"accuracy": r["accuracy"], "suite": suite}

    # --- ExponentiatedGradient (in) ---
    eg_results = apply_exponentiated_gradient(X, y, race_values, ARTIFACT_DIR)
    eg_metrics = {}
    for algorithm, r in eg_results.items():
        y_test = r["y_test"].to_numpy()
        pmf = r["model"]._pmf_predict(r["X_test"])
        y_prob = pmf[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        suite = run_full_metric_suite(
            y_true=y_test, y_pred=y_pred, y_prob=y_prob,
            protected_attr=r["test_protected_attr"], n_bootstrap=500,
        )
        eg_metrics[algorithm] = {"accuracy": r["accuracy"], "suite": suite}

    # --- ThresholdOptimizer (post) ---
    to_results = apply_threshold_optimizer(X, y, race_values, ARTIFACT_DIR)
    to_metrics = {}
    for algorithm, r in to_results.items():
        y_test = r["y_test"].to_numpy()
        np.random.seed(42)  # reproducibility, matching apply_threshold_optimizer's internal seeding
        y_pred = r["model"].predict(r["X_test"], sensitive_features=r["test_protected_attr"])
        suite = run_full_metric_suite(
            y_true=y_test, y_pred=y_pred, y_prob=y_pred.astype(float),
            protected_attr=r["test_protected_attr"], n_bootstrap=500,
        )
        to_metrics[algorithm] = {"accuracy": r["accuracy"], "suite": suite}

    # --- Persist, with dedupe check per (audit_run, method) ---
    method_data = {
        "baseline": (baseline_results, "none", None),
        "reweighing": (reweighed_metrics, "pre", None),
        "exponentiated_gradient": (eg_metrics, "in",
            "Computed via Fairlearn's internal _pmf_predict thresholded at 0.5 "
            "(deterministic approximation), not the stochastic .predict()."),
        "threshold_optimizer": (to_metrics, "post",
            "Not applicable — ThresholdOptimizer outputs group-specific hard "
            "decisions by design, not probability scores."),
    }

    for algorithm in ["logistic_regression", "gradient_boosting"]:
        model_row = db.query(MLModel).filter(MLModel.algorithm == algorithm).first()
        if not model_row:
            print(f"WARNING: no MLModel row found for {algorithm}, skipping")
            continue

        audit_run = get_or_create_audit_run(db, model_row)

        for method, (results_dict, stage_type, calibration_note) in method_data.items():
            if already_persisted(db, audit_run.id, method):
                print(f"SKIP: {algorithm}/{method} already persisted for audit_run={audit_run.id}")
                continue

            result = results_dict[algorithm]
            db.add(MitigationResult(
                audit_run_id=audit_run.id,
                method=method,
                stage_type=stage_type,
                accuracy=result["accuracy"],
                fairness_metrics=strip_cis_to_point_estimates(result["suite"], calibration_note),
            ))
            print(f"Persisted {algorithm}/{method}")

    db.commit()
    db.close()
    print("Done.")


if __name__ == "__main__":
    main()