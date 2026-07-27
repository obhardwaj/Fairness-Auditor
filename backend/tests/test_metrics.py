"""
Regression test: validates the metric suite against ProPublica's own
published analysis, which audited COMPAS's decile_score directly (not a
downstream trained classifier). This is the correct external ground truth
to check your metric *code* is correct, independent of your own models.

Published figures (ProPublica, "Machine Bias", 2016): among defendants who
did NOT reoffend, Black defendants were flagged high-risk at ~45%, white
defendants at ~23% — a roughly 20+ percentage point false positive rate gap.
"""
import pandas as pd
import numpy as np
import pytest

from app.ml.compas_pipeline import load_and_filter_compas
from app.ml.metrics import false_positive_rate_difference, false_negative_rate_difference


@pytest.fixture
def compas_race_binary():
    df = load_and_filter_compas("data/compas-scores-two-years.csv")
    # Restrict to the two groups ProPublica's headline analysis focused on
    df = df[df["race"].isin(["African-American", "Caucasian"])]

    y_true = df["two_year_recid"].to_numpy()
    y_pred = (df["decile_score"] >= 5).to_numpy().astype(int)  # "high risk" per COMPAS's own scale
    protected_attr = df["race"].to_numpy()
    return y_true, y_pred, protected_attr


def test_false_positive_rate_gap_matches_published_range(compas_race_binary):
    y_true, y_pred, protected_attr = compas_race_binary
    fpr_diff = false_positive_rate_difference(y_true, y_pred, protected_attr)

    # ProPublica reported ~45% vs ~23% -> roughly a 0.20-0.25 gap.
    # Allow a reasonably wide band since exact filtering/thresholding can shift it slightly.
    assert 0.15 <= fpr_diff <= 0.30, f"FPR gap {fpr_diff:.3f} outside expected published range"


def test_false_negative_rate_gap_matches_published_direction(compas_race_binary):
    y_true, y_pred, protected_attr = compas_race_binary
    fnr_diff = false_negative_rate_difference(y_true, y_pred, protected_attr)

    # ProPublica found the FNR gap runs the opposite direction (white defendants
    # under-flagged relative to Black defendants) — just check it's non-trivial.
    assert fnr_diff >= 0.10, f"FNR gap {fnr_diff:.3f} smaller than expected"