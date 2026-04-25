"""SVM RBF con configuración fiel al paper: γ = 1/n_features, C = 1, sin tuning."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def make_svm(n_features: int, C: float = 1.0) -> SVC:
    return SVC(kernel="rbf", C=C, gamma=1.0 / max(n_features, 1), probability=True)


def fit_svm_pipeline(
    X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0
) -> tuple[MinMaxScaler, SVC]:
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_s = scaler.transform(X_train)
    clf = make_svm(n_features=X_s.shape[1], C=C)
    clf.fit(X_s, y_train)
    return scaler, clf


def predict_proba_pipeline(
    scaler: MinMaxScaler, clf: SVC, X: np.ndarray
) -> np.ndarray:
    return clf.predict_proba(scaler.transform(X))


def aggregate_subject_proba(
    probas: np.ndarray, subject_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Promedia softmax por sujeto. Devuelve (subjects, probs)."""
    uniq = np.unique(subject_ids)
    out = np.zeros((len(uniq), probas.shape[1]))
    for i, s in enumerate(uniq):
        out[i] = probas[subject_ids == s].mean(axis=0)
    return uniq, out
