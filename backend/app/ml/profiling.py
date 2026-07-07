"""
Week 1 deliverable: data profiling.

Given a raw dataframe, detect:
1. Explicitly protected attributes (matched against a known list of common
   protected-attribute column names).
2. Proxy variables — features that correlate strongly with a protected
   attribute and could leak protected information indirectly (e.g. zip code
   as a proxy for race).

This is intentionally simple to start (Week 1). Extend later with more
robust proxy detection (e.g. mutual information instead of raw correlation,
categorical association via Cramer's V).
"""
from __future__ import annotations

import pandas as pd
import numpy as np

# Common protected-attribute column name patterns to match against (case-insensitive).
KNOWN_PROTECTED_PATTERNS = [
    "race", "ethnicity", "sex", "gender", "age", "religion",
    "disability", "national_origin", "marital_status",
]

PROXY_CORRELATION_THRESHOLD = 0.5


def detect_protected_attributes(df: pd.DataFrame) -> list[str]:
    """Match column names against known protected-attribute patterns."""
    detected = []
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in KNOWN_PROTECTED_PATTERNS):
            detected.append(col)
    return detected


def detect_proxies(
    df: pd.DataFrame,
    protected_attributes: list[str],
    threshold: float = PROXY_CORRELATION_THRESHOLD,
) -> dict[str, float]:
    """
    For each non-protected numeric/categorical column, compute an association
    score with each protected attribute. Flag columns above `threshold`.

    NOTE: this is a Week 1 stub using simple correlation for numeric columns
    and a placeholder for categorical association. Replace with mutual
    information / Cramer's V for a more defensible Week 2+ implementation.
    """
    proxies: dict[str, float] = {}
    candidate_cols = [c for c in df.columns if c not in protected_attributes]

    for protected_col in protected_attributes:
        if protected_col not in df.columns:
            continue
        protected_series = df[protected_col]
        if not pd.api.types.is_numeric_dtype(protected_series):
            protected_series = protected_series.astype("category").cat.codes

        for candidate in candidate_cols:
            series = df[candidate]
            if not pd.api.types.is_numeric_dtype(series):
                series = series.astype("category").cat.codes
            try:
                corr = np.corrcoef(protected_series, series)[0, 1]
            except Exception:
                continue
            if abs(corr) >= threshold:
                proxies[candidate] = round(float(abs(corr)), 3)

    return proxies


def profile_dataset(df: pd.DataFrame) -> dict:
    """Run full Week 1 profiling and return a summary dict to persist on the Dataset row."""
    protected_attrs = detect_protected_attributes(df)
    proxies = detect_proxies(df, protected_attrs)
    return {
        "protected_attributes": protected_attrs,
        "detected_proxies": proxies,
        "row_count": len(df),
        "columns": list(df.columns),
    }
