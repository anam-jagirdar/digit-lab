"""Data loading, candidate models and model selection."""
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


def load_data(test_size: float = 0.25, seed: int = SEED):
    """8x8 grayscale digits, flattened to 64 features. Stratified so every digit is in both splits."""
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=test_size, stratify=digits.target, random_state=seed
    )
    return x_train, x_test, y_train, y_test


def candidate_models(seed: int = SEED) -> dict:
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "k-nearest neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        "Random forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "SVM (RBF kernel)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
    }


@dataclass
class CVResult:
    name: str
    mean: float
    std: float


def compare_models(models: dict, x, y, folds: int = 5, seed: int = SEED) -> list[CVResult]:
    """Cross-validate on the training data only, so the test set stays untouched until the end."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, x, y, cv=cv)
        results.append(CVResult(name, float(np.mean(scores)), float(np.std(scores))))
    return sorted(results, key=lambda r: r.mean, reverse=True)
