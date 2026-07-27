"""
Mitigation strategies for fairness violations.

Three methods, one per stage of the ML pipeline, matching the roadmap:
- Pre-processing: reweighing (this file, implemented first)
- In-processing: ExponentiatedGradient (Fairlearn) — next
- Post-processing: ThresholdOptimizer (Fairlearn) — after that

Each mitigation function returns a fitted model plus the reweighted/adjusted
predictions, so the same run_full_metric_suite() from Week 2 can be re-run
against the output to measure improvement.
"""
from __future__ import annotations

import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from pandas.core.common import random_state

from app.ml.baseline_models import MODEL_REGISTRY
from app.ml.compas_pipeline import load_and_filter_compas, filter_to_binary_race, build_feature_matrix
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split


def apply_reweighing(X, y, protected_attr_col_name: str, protected_attr_values, artifact_dir: str):
    """
    Computes per-sample weights using AIF360's Reweighing algorithm, then
    trains a fresh classifier using those weights (sample_weight param).
    """
    # AIF360's BinaryLabelDataset requires every column to be numeric, so the
    # protected attribute must be encoded before construction — even though
    # we want to reason about it as "African-American" vs "Caucasian" etc.
    unique_groups = np.unique(protected_attr_values)
    if len(unique_groups) != 2:
        raise ValueError(
            f"Reweighing here assumes exactly 2 groups, got {len(unique_groups)}: {unique_groups}. "
            "Filter to a binary comparison first (e.g. two largest race categories)."
        )

    # Map string labels -> 0/1 codes. group_to_code[unique_groups[0]] = 0, etc.
    group_to_code = {group: code for code, group in enumerate(unique_groups)}
    protected_attr_codes = np.array([group_to_code[g] for g in protected_attr_values])

    df_for_aif = X.copy()
    df_for_aif["__label__"] = y.to_numpy() if hasattr(y, "to_numpy") else y
    df_for_aif["__protected__"] = protected_attr_codes  # <-- now numeric

    privileged_code = group_to_code[unique_groups[0]]
    unprivileged_code = group_to_code[unique_groups[1]]

    aif_dataset = BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=df_for_aif,
        label_names=["__label__"],
        protected_attribute_names=["__protected__"],
    )

    privileged_groups = [{"__protected__": privileged_code}]
    unprivileged_groups = [{"__protected__": unprivileged_code}]

    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    reweighed_dataset = rw.fit_transform(aif_dataset)

    sample_weights = reweighed_dataset.instance_weights

    # --- everything below this line is unchanged from before ---
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    for algorithm, model_fn in MODEL_REGISTRY.items():
        clf = model_fn()
        clf.fit(X_train, y_train, sample_weight=w_train)
        preds = clf.predict(X_test)
        acc = float((preds == y_test.to_numpy()).mean())

        artifact_path = f"{artifact_dir}/reweighed_{algorithm}.joblib"
        import joblib
        joblib.dump(clf, artifact_path)

        results[algorithm] = {
            "model": clf,
            "accuracy": acc,
            "artifact_path": artifact_path,
            "X_test": X_test,
            "y_test": y_test,
            "test_protected_attr": protected_attr_values[X_test.index] if hasattr(protected_attr_values, "__getitem__") else None,
        }

    return results

def run_reweighing_on_race(csv_path: str, artifact_dir: str):
    """
    End-to-end: load COMPAS, filter to binary race, build features, apply
    reweighing, train, and return results — mirrors the shape of
    run_compas_baseline_pipeline() from Week 1 so the API layer can persist
    it the same way.
    """
    df = load_and_filter_compas(csv_path)
    df = filter_to_binary_race(df)  # <-- the new step

    X, y, protected_attr_cols, protected_attrs_raw = build_feature_matrix(df)
    race_values = protected_attrs_raw["race"].to_numpy()

    results = apply_reweighing(
        X=X, y=y,
        protected_attr_col_name="race",
        protected_attr_values=race_values,
        artifact_dir=artifact_dir,
    )
    return results

def apply_exponentiated_gradient(X, y, protected_attr_values, artifact_dir: str, constraint_fn=None):
    if constraint_fn is None:
        constraint_fn = lambda: DemographicParity()

    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X, y, protected_attr_values, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    for algorithm, model_fn in MODEL_REGISTRY.items():
        base_estimator = model_fn()
        constraint = constraint_fn()

        mitigator = ExponentiatedGradient(estimator=base_estimator, constraints=constraint)
        mitigator.fit(X_train, y_train, sensitive_features=prot_train)

        # Use the deterministic weighted-average PMF instead of the stochastic
        # .predict() — ExponentiatedGradient's .predict() samples randomly from
        # its classifier mixture each call, which made results non-reproducible
        # across runs. _pmf_predict gives the same expected-value output every
        # time for a given fitted mitigator.
        pmf = mitigator._pmf_predict(X_test)
        y_prob = pmf[:, 1]
        preds = (y_prob >= 0.5).astype(int)

        acc = float((preds == y_test.to_numpy()).mean())

        artifact_path = f"{artifact_dir}/expgrad_{algorithm}.joblib"
        import joblib
        joblib.dump(mitigator, artifact_path)

        results[algorithm] = {
            "model": mitigator,
            "accuracy": acc,
            "artifact_path": artifact_path,
            "X_test": X_test,
            "y_test": y_test,
            "test_protected_attr": prot_test,
        }

    return results


def apply_threshold_optimizer(X, y, protected_attr_values, artifact_dir: str, constraint: str = "demographic_parity"):
    """
    Post-processing mitigation: ThresholdOptimizer takes an ALREADY-TRAINED
    model and adjusts decision thresholds per group to satisfy a fairness
    constraint — it doesn't retrain anything, just recalibrates cutoffs.

    Trains a fresh baseline model per algorithm first (since ThresholdOptimizer
    needs a fitted estimator to wrap), then wraps it.
    """
    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X, y, protected_attr_values, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    for algorithm, model_fn in MODEL_REGISTRY.items():
        base_estimator = model_fn()
        base_estimator.fit(X_train, y_train)

        postprocessor = ThresholdOptimizer(
            estimator=base_estimator,
            constraints=constraint,
            predict_method="predict_proba",
            prefit=True,
        )
        postprocessor.fit(X_train, y_train, sensitive_features=prot_train)

        import numpy as np
        np.random.seed(42)
        preds = postprocessor.predict(X_test, sensitive_features=prot_test)
        acc = float((preds == y_test.to_numpy()).mean())

        artifact_path = f"{artifact_dir}/thresholdopt_{algorithm}.joblib"
        import joblib
        joblib.dump(postprocessor, artifact_path)

        results[algorithm] = {
            "model": postprocessor,
            "accuracy": acc,
            "artifact_path": artifact_path,
            "X_test": X_test,
            "y_test": y_test,
            "test_protected_attr": prot_test,
        }

    return results

