"""
Week 2 deliverable (stub for now): fairness metric suite with bootstrap CIs.

Planned functions:
- demographic_parity_difference(y_pred, protected_attr) -> (value, ci_lower, ci_upper)
- equalized_odds_difference(y_true, y_pred, protected_attr) -> (...)
- equal_opportunity_difference(y_true, y_pred, protected_attr) -> (...)
- disparate_impact_ratio(y_pred, protected_attr) -> (...)
- calibration_within_groups(y_true, y_prob, protected_attr) -> per-group calibration curve data

Use fairlearn.metrics for the point estimates and wrap each in a bootstrap
resampling loop (scipy.stats.bootstrap or a manual loop) to get confidence
intervals — this is the differentiator vs. typical student fairness projects
that report a single point estimate with no uncertainty quantification.

Validate against ProPublica's published COMPAS numbers as a regression test
in tests/test_metrics.py before trusting this on other datasets.
"""

# TODO(week 2): implement using fairlearn.metrics as the computational backend
# from fairlearn.metrics import MetricFrame, demographic_parity_difference as dpd
