import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
        

class ContradictionMetric():
    """Computes the contradiction score along an experiment

    We call contradiction the difference between predictions of two successive
    models on an isolated test set.

    Args:
        mode: Computation mode, can be "auto" (default), soft, hard
    """
    def __init__(self, mode='auto'):
        self.mode = mode
        self.reset()

    """Stores contradiction for a new iteration.

    Args:
        probas_test : Predictions of shape (n_samples, n_classes)
    """
    def update(self, probas_test: np.array):
        prev_probas_test = self._cache_probas_test
        self._cache_probas_test = probas_test

        if prev_probas_test is None:
            return

        if self.mode == 'auto':
            mode = 'soft' if probas_test.shape[0] < 1000 else 'hard'

        if mode == 'soft':
            self._value = np.abs(prev_probas_test - probas_test).mean()
        elif mode == 'hard':
            self._value = (np.argmax(prev_probas_test, axis=1) == np.argmax(probas_test, axis=1)).mean()


    """Returns the recorded metrics
    """
    def get(self):
        return self._value


    """Reset the metrics for a new experiment
    """
    def reset(self):
        self._cache_probas_test = None


def exploration_score(X_selected: np.ndarray, X_test: np.ndarray) -> float:
    """Compute the nearest neighbor based exploration score.

    Args:
        X_selected: Samples selected so far
        X_test: Left out test data

    Returns:
        Mean distance between test samples and their closest selected neighbor
    """
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_train)
    return nn.kneighbors(X_test, n_neighbors=1)[0].mean()


def exploration_score_from_knn(
        knn: KNeighborsClassifier, X_test: np.ndarray) -> float:
    """Compute the nearest neighbor based exploration score.

    Args:
        knn: KNearestClassifier trained on samples selected so far
        X_test: Left out test data

    Returns:
        Mean distance between test samples and their closest selected neighbor
    """
    return knn.kneighbors(X_test, n_neighbors=1)[0].mean()


def classifier_agreement(clf_a, clf_b, X_batch: np.ndarray) -> float:
    """Compute the agreement score (similar prediction ratio)

    Args:
        clf_a: First classifier, usually task specific
        clf_b: Second classifier, usually 1-nearest-neighbor
        X_batch: Set of samples on which agreement is measured

    Returns:
        Agreement between both classifier on the sample batch.
    """
    return (clf_a.predict(X_batch) == clf_b.predict(X_batch)).mean()