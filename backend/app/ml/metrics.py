"""
Fairness metric suite with bootstrap confidence intervals.

Design principle: each metric function takes raw arrays (y_true, y_pred,
protected_attr) and returns a plain float or dict — no DB/ORM dependency here.
The API/agent layer is responsible for persisting results into MetricResult rows.
"""
from __future__ import annotations

import numpy as np
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference as fl_demographic_parity_difference,
    equalized_odds_difference as fl_equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)
from sklearn.calibration import calibration_curve


# ---------------------------------------------------------------------------
# Point-estimate metrics (thin wrappers over Fairlearn — don't reimplement
# these by hand, Fairlearn's implementations are the ones the field cites)
# ---------------------------------------------------------------------------

def demographic_parity_difference(y_pred, protected_attr) -> float:
    return float(fl_demographic_parity_difference(
        y_true=np.zeros_like(y_pred),  # unused by this metric, required by signature
        y_pred=y_pred,
        sensitive_features=protected_attr,
    ))


def disparate_impact_ratio(y_pred, protected_attr) -> float:
    """Ratio of the lowest group selection rate to the highest (the '80% rule')."""
    mf = MetricFrame(metrics=selection_rate, y_true=np.zeros_like(y_pred),
                      y_pred=y_pred, sensitive_features=protected_attr)
    rates = mf.by_group
    return float(rates.min() / rates.max())


def equalized_odds_difference(y_true, y_pred, protected_attr) -> float:
    return float(fl_equalized_odds_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=protected_attr
    ))


def false_positive_rate_difference(y_true, y_pred, protected_attr) -> float:
    mf = MetricFrame(metrics=false_positive_rate, y_true=y_true,
                      y_pred=y_pred, sensitive_features=protected_attr)
    return float(mf.by_group.max() - mf.by_group.min())


def false_negative_rate_difference(y_true, y_pred, protected_attr) -> float:
    mf = MetricFrame(metrics=false_negative_rate, y_true=y_true,
                      y_pred=y_pred, sensitive_features=protected_attr)
    return float(mf.by_group.max() - mf.by_group.min())


def calibration_within_groups(y_true, y_prob, protected_attr, n_bins: int = 10) -> dict:
    """
    Per-group calibration curve. Returns {group_name: (prob_true, prob_pred)}
    for plotting — not a single scalar, since calibration is inherently a curve.
    """
    results = {}
    protected_attr = np.asarray(protected_attr)
    for group in np.unique(protected_attr):
        mask = protected_attr == group
        if mask.sum() < n_bins:  # not enough samples in this group to bin meaningfully
            continue
        prob_true, prob_pred = calibration_curve(
            y_true[mask], y_prob[mask], n_bins=n_bins, strategy="quantile"
        )
        results[str(group)] = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals — generic wrapper around any metric function
# ---------------------------------------------------------------------------

def bootstrap_ci(
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
    **arrays,
) -> tuple[float, float, float]:
    """
    Generic bootstrap wrapper. `arrays` should be the keyword arguments the
    metric_fn expects (e.g. y_true=..., y_pred=..., protected_attr=...) —
    all arrays must be the same length; they're resampled together (same
    indices) to preserve row-wise correspondence.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(random_state)
    keys = list(arrays.keys())
    n = len(arrays[keys[0]])
    arrays = {k: np.asarray(v) for k, v in arrays.items()}

    point_estimate = metric_fn(**arrays)

    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = {k: v[idx] for k, v in arrays.items()}
        try:
            estimate = metric_fn(**resampled)
        except Exception:
            continue  # skip resamples that fail (e.g. a group disappears entirely)
        bootstrap_estimates.append(estimate)

    alpha = (1 - ci) / 2
    lower = float(np.percentile(bootstrap_estimates, 100 * alpha))
    upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha)))

    return point_estimate, lower, upper


def run_full_metric_suite(y_true, y_pred, y_prob, protected_attr, n_bootstrap: int = 1000) -> dict:
    """
    Convenience function: runs the full suite with bootstrap CIs for a single
    protected attribute. Call once per protected attribute (e.g. once for
    'race', once for 'sex') since group definitions differ each time.
    """
    suite = {}

    suite["demographic_parity_difference"] = bootstrap_ci(
        lambda y_pred, protected_attr: demographic_parity_difference(y_pred, protected_attr),
        n_bootstrap=n_bootstrap, y_pred=y_pred, protected_attr=protected_attr,
    )
    suite["disparate_impact_ratio"] = bootstrap_ci(
        lambda y_pred, protected_attr: disparate_impact_ratio(y_pred, protected_attr),
        n_bootstrap=n_bootstrap, y_pred=y_pred, protected_attr=protected_attr,
    )
    suite["equalized_odds_difference"] = bootstrap_ci(
        lambda y_true, y_pred, protected_attr: equalized_odds_difference(y_true, y_pred, protected_attr),
        n_bootstrap=n_bootstrap, y_true=y_true, y_pred=y_pred, protected_attr=protected_attr,
    )
    suite["false_positive_rate_difference"] = bootstrap_ci(
        lambda y_true, y_pred, protected_attr: false_positive_rate_difference(y_true, y_pred, protected_attr),
        n_bootstrap=n_bootstrap, y_true=y_true, y_pred=y_pred, protected_attr=protected_attr,
    )
    suite["false_negative_rate_difference"] = bootstrap_ci(
        lambda y_true, y_pred, protected_attr: false_negative_rate_difference(y_true, y_pred, protected_attr),
        n_bootstrap=n_bootstrap, y_true=y_true, y_pred=y_pred, protected_attr=protected_attr,
    )

    # Calibration isn't bootstrapped here — it's a curve, not a scalar. Bootstrap
    # CIs on calibration curves are possible but add complexity; revisit if time allows.
    suite["calibration_within_groups"] = calibration_within_groups(y_true, y_prob, protected_attr)

    return suite