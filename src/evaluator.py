"""
Linear probe evaluator for active learning experiments.

Follows the "Fully Supervised with Self-Supervised Embedding" framework
described in Section 4.2.2 of Hacohen et al. (2022):
a logistic regression classifier is trained on the labeled embeddings
and evaluated on the held-out test embeddings.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def evaluate_linear_probe(
    train_features: np.ndarray,
    train_labels:   np.ndarray,
    test_features:  np.ndarray,
    test_labels:    np.ndarray,
    seed: int = 42,
) -> float:
    """
    Fit a logistic regression probe on labeled embeddings and score on test set.

    Parameters
    ----------
    train_features : np.ndarray  shape (N_train, D)
        L2-normalised embeddings for the currently labeled examples.
    train_labels : np.ndarray  shape (N_train,)
        Ground-truth class indices for the labeled examples.
    test_features : np.ndarray  shape (N_test, D)
        L2-normalised embeddings for the full test set.
    test_labels : np.ndarray  shape (N_test,)
        Ground-truth class indices for the test set.
    seed : int
        Random state passed to LogisticRegression for reproducibility.

    Returns
    -------
    float
        Top-1 accuracy on the test set (value in [0, 1]).
    """
    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver='lbfgs',
    )
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    return float(accuracy_score(test_labels, predictions))
