"""
Week 1 deliverable: baseline classifier training.

Trains standard classifiers on a prepared (X, y) split so there's a realistic
model to audit in later weeks. Kept deliberately simple — the fairness
analysis, not baseline accuracy, is the point of this project.
"""
from __future__ import annotations

import joblib
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@dataclass
class TrainedModel:
    name: str
    algorithm: str
    model: object
    accuracy: float
    artifact_path: str


MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "gradient_boosting": lambda: GradientBoostingClassifier(),
}


def train_baseline_model(
    X, y, algorithm: str, artifact_dir: str, model_name: str, test_size: float = 0.2, random_state: int = 42
) -> TrainedModel:
    if algorithm not in MODEL_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Options: {list(MODEL_REGISTRY)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = MODEL_REGISTRY[algorithm]()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    artifact_path = f"{artifact_dir}/{model_name}.joblib"
    joblib.dump(clf, artifact_path)

    return TrainedModel(
        name=model_name,
        algorithm=algorithm,
        model=clf,
        accuracy=acc,
        artifact_path=artifact_path,
    )