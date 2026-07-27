"""
One-off script: computes baseline + reweighed fairness metrics for both
logistic_regression and gradient_boosting on the race-binary COMPAS subset,
and persists all four as MitigationResult rows.

Run with: docker compose exec api python scripts/persist_mitigation_comparison.py
"""
import joblib

from app.core.db import SessionLocal
from app.models.models import AuditRun, AuditStatus, MitigationResult, MLModel, Dataset
from app.ml.compas_pipeline import load_and_filter_compas, filter_to_binary_race, build_feature_matrix
from app.ml.metrics import run_full_metric_suite
from app.ml.mitigation import apply_reweighing
from app.ml.mitigation import apply_reweighing, apply_exponentiated_gradient, apply_threshold_optimizer

CSV_PATH = "data/compas-scores-two-years.csv"
ARTIFACT_DIR = "data/artifacts"


def get_or_create_audit_run(db, model_row) -> AuditRun:
    """Reuse an existing audit run for this model if one exists, else create one."""
    existing = db.query(AuditRun).filter(AuditRun.model_id == model_row.id).first()
    if existing:
        return existing
    audit_run = AuditRun(model_id=model_row.id, status=AuditStatus.completed)
    db.add(audit_run)
    db.commit()
    db.refresh(audit_run)
    return audit_run


def strip_cis_to_point_estimates(suite: dict, calibration_note: str | None = None) -> dict:
    """
    MitigationResult.fairness_metrics is a JSON snapshot — flatten (point, lo, hi)
    tuples. If calibration_note is provided, store an explicit null + reason
    instead of silently omitting the key (e.g. for hard-decision methods like
    ThresholdOptimizer where calibration isn't a meaningful concept).
    """
    out = {}
    for metric_name, result in suite.items():
        if metric_name == "calibration_within_groups":
            continue
        point, lower, upper = result
        out[metric_name] = {"value": point, "ci_lower": lower, "ci_upper": upper}

    if calibration_note is not None:
        out["calibration_within_groups"] = {
            "value": None,
            "note": calibration_note,
        }

    return out


def main():
    db = SessionLocal()

    df = load_and_filter_compas(CSV_PATH)
    df = filter_to_binary_race(df)
    X, y, protected_attr_cols, protected_attrs_raw = build_feature_matrix(df)
    race_values = protected_attrs_raw["race"].to_numpy()

    # --- Baseline metrics for both algorithms (on race-binary subset) ---
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

    # --- Reweighing (pre-processing) ---
    reweigh_results = apply_reweighing(
        X=X, y=y, protected_attr_col_name="race",
        protected_attr_values=race_values, artifact_dir=ARTIFACT_DIR,
    )
    reweighed_metrics = {}
    for algorithm, r in reweigh_results.items():
        model = r["model"]
        X_test, y_test = r["X_test"], r["y_test"].to_numpy()
        protected_test = r["test_protected_attr"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        suite = run_full_metric_suite(
            y_true=y_test, y_pred=y_pred, y_prob=y_prob,
            protected_attr=protected_test, n_bootstrap=500,
        )
        reweighed_metrics[algorithm] = {"accuracy": r["accuracy"], "suite": suite}

    # --- ExponentiatedGradient (in-processing) ---
    eg_results = apply_exponentiated_gradient(X, y, race_values, ARTIFACT_DIR)
    eg_metrics = {}
    for algorithm, r in eg_results.items():
        y_test = r["y_test"].to_numpy()
        # Using Fairlearn's internal _pmf_predict since ExponentiatedGradient
        # has no public predict_proba.
        pmf = r["model"]._pmf_predict(r["X_test"])
        y_prob = pmf[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        suite = run_full_metric_suite(
            y_true=y_test, y_pred=y_pred, y_prob=y_prob,
            protected_attr=r["test_protected_attr"], n_bootstrap=500,
        )
        eg_metrics[algorithm] = {"accuracy": r["accuracy"], "suite": suite}

    # --- ThresholdOptimizer (post-processing) ---
    to_results = apply_threshold_optimizer(X, y, race_values, ARTIFACT_DIR)
    to_metrics = {}
    for algorithm, r in to_results.items():
        y_test = r["y_test"].to_numpy()
        y_pred = r["model"].predict(r["X_test"], sensitive_features=r["test_protected_attr"])
        suite = run_full_metric_suite(
            y_true=y_test, y_pred=y_pred, y_prob=y_pred.astype(float),
            protected_attr=r["test_protected_attr"], n_bootstrap=500,
        )
        to_metrics[algorithm] = {"accuracy": r["accuracy"], "suite": suite}

    # --- Persist all rows: baseline, reweighing, exponentiated_gradient, threshold_optimizer ---
    for algorithm in ["logistic_regression", "gradient_boosting"]:
        model_row = db.query(MLModel).filter(MLModel.algorithm == algorithm).first()
        if not model_row:
            print(f"WARNING: no MLModel row found for {algorithm}, skipping")
            continue

        audit_run = get_or_create_audit_run(db, model_row)

        baseline = baseline_results[algorithm]
        db.add(MitigationResult(
            audit_run_id=audit_run.id, method="baseline", stage_type="none",
            accuracy=baseline["accuracy"],
            fairness_metrics=strip_cis_to_point_estimates(baseline["suite"]),
        ))

        reweighed = reweighed_metrics[algorithm]
        db.add(MitigationResult(
            audit_run_id=audit_run.id, method="reweighing", stage_type="pre",
            accuracy=reweighed["accuracy"],
            fairness_metrics=strip_cis_to_point_estimates(reweighed["suite"]),
        ))

        eg = eg_metrics[algorithm]
        db.add(MitigationResult(
            audit_run_id=audit_run.id, method="exponentiated_gradient", stage_type="in",
            accuracy=eg["accuracy"],
            fairness_metrics=strip_cis_to_point_estimates(eg["suite"]),
        ))

        to = to_metrics[algorithm]
        db.add(MitigationResult(
            audit_run_id=audit_run.id, method="threshold_optimizer", stage_type="post",
            accuracy=to["accuracy"],
            fairness_metrics=strip_cis_to_point_estimates(
                to["suite"],
                calibration_note="Not applicable — ThresholdOptimizer outputs group-specific hard decisions by design, not probability scores.",
            ),
        ))

        print(f"Persisted 4 rows (baseline, reweighing, exp_gradient, threshold_opt) for {algorithm}")

    db.commit()
    db.close()
    print("Done.")


if __name__ == "__main__":
    main()