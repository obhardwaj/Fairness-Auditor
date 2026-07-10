"""
Loads the raw ProPublica COMPAS CSV, applies the standard filtering steps
used in the fairness literature (so results are comparable to published
ProPublica/Fairlearn benchmarks), and trains baseline classifiers on it.

Standard filters applied (matching ProPublica's own analysis):
- Drop rows where days_b_screening_arrest is missing, > 30, or < -30
  (screening happened too far from the arrest date to be reliable)
- Drop rows where is_recid == -1 (recidivism outcome unknown)
- Drop rows where c_charge_degree == 'O' (ordinary traffic offenses,
  not eligible for jail time, shouldn't be in a recidivism model)
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.ml.baseline_models import train_baseline_model, TrainedModel


RAW_COLUMNS_NEEDED = [
    "sex", "age", "race", "juv_fel_count", "juv_misd_count", "juv_other_count",
    "priors_count", "days_b_screening_arrest", "c_charge_degree", "is_recid",
    "two_year_recid",
]


def load_and_filter_compas(csv_path: str) -> pd.DataFrame:
    """Load the raw COMPAS CSV and apply ProPublica's standard filtering steps."""
    df = pd.read_csv(csv_path)

    # De-duplicate the repeated decile_score / priors_count columns pandas
    # suffixes on load (e.g. 'priors_count.1') — keep only the first occurrence.
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop rows with missing or out-of-range screening window.
    df = df[df["days_b_screening_arrest"].notna()]
    df = df[(df["days_b_screening_arrest"] <= 30) & (df["days_b_screening_arrest"] >= -30)]

    # Drop rows with unknown recidivism outcome.
    df = df[df["is_recid"] != -1]

    # Drop ordinary traffic offenses (not jail-eligible, shouldn't be modeled).
    df = df[df["c_charge_degree"] != "O"]

    df = df.reset_index(drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build (X, y, protected_attribute_columns) from the filtered COMPAS dataframe.

    Keeps race and sex in X (encoded) so they remain available for the Week 2
    fairness metric suite to slice on — do NOT drop protected attributes
    entirely, since you need them downstream to compute group-wise metrics.
    """
    features = df[[
        "sex", "age", "race", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "priors_count", "c_charge_degree",
    ]].copy()

    # Simple label-encode categoricals for the baseline models. Keep a copy of
    # the original (unencoded) race/sex columns separately — the fairness
    # metric suite in Week 2 should slice on the human-readable category,
    # not the integer code.
    protected_attrs_raw = df[["race", "sex"]].copy()

    encoders = {}
    for col in ["sex", "race", "c_charge_degree"]:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        encoders[col] = le

    y = df["two_year_recid"]

    return features, y, ["race", "sex"], protected_attrs_raw


def run_compas_baseline_pipeline(csv_path: str, artifact_dir: str) -> dict:
    """
    End-to-end: load, filter, build features, train both baseline models.
    Returns a dict with everything needed to persist Dataset + MLModel rows.
    """
    import os
    os.makedirs(artifact_dir, exist_ok=True)

    original_row_count = len(pd.read_csv(csv_path))
    df = load_and_filter_compas(csv_path)
    X, y, protected_attr_cols, protected_attrs_raw = build_feature_matrix(df)

    trained_models = []
    for algorithm, model_name in [
        ("logistic_regression", "compas_logistic_regression_v1"),
        ("gradient_boosting", "compas_gradient_boosting_v1"),
    ]:
        trained = train_baseline_model(
            X=X, y=y, algorithm=algorithm,
            artifact_dir=artifact_dir, model_name=model_name,
        )
        trained_models.append(trained)
        print(f"{algorithm}: accuracy={trained.accuracy:.4f}  ->  {trained.artifact_path}")

    return {
        "row_count_original": original_row_count,
        "row_count_after_filtering": len(df),
        "protected_attributes": protected_attr_cols,
        "trained_models": trained_models,  # list[TrainedModel], used by the API layer to persist rows
    }


if __name__ == "__main__":
    # Quick manual run: python -m app.ml.compas_pipeline
    import os
    os.makedirs("data/artifacts", exist_ok=True)
    summary = run_compas_baseline_pipeline(
        csv_path="data/compas-scores-two-years.csv",
        artifact_dir="data/artifacts",
    )
    print(summary)