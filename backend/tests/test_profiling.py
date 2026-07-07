import pandas as pd

from app.ml.profiling import detect_protected_attributes, detect_proxies, profile_dataset


def test_detect_protected_attributes_matches_known_patterns():
    df = pd.DataFrame({
        "race": ["A", "B"],
        "income": [50000, 60000],
        "zip_code": ["12345", "12346"],
    })
    detected = detect_protected_attributes(df)
    assert "race" in detected
    assert "income" not in detected


def test_detect_proxies_flags_correlated_column():
    df = pd.DataFrame({
        "race_code": [0, 0, 1, 1, 0, 1],
        "zip_code_code": [0, 0, 1, 1, 0, 1],  # perfectly correlated with race_code
        "unrelated": [5, 2, 9, 1, 7, 3],
    })
    proxies = detect_proxies(df, protected_attributes=["race_code"], threshold=0.5)
    assert "zip_code_code" in proxies
    assert "unrelated" not in proxies


def test_profile_dataset_returns_expected_keys():
    df = pd.DataFrame({"sex": ["M", "F"], "score": [1, 2]})
    profile = profile_dataset(df)
    assert set(profile.keys()) == {
        "protected_attributes", "detected_proxies", "row_count", "columns"
    }

# TODO(week 2): add tests/test_metrics.py comparing computed fairness metrics
# on the COMPAS dataset against ProPublica's published numbers as a
# regression check before trusting the metric suite on other datasets.
