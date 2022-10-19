# Note: This code is inspired from modAL implementation
# https://modal-python.readthedocs.io/en/latest/content/query_strategies/ranked_batch_mode.html

import numpy as np

from .version import check_modules
check_modules('sklearn', 'batch')  # noqa

from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

from .base import BaseQuerySampler


class RankedBatchSampler(BaseQuerySampler):
    """Selects samples to label by maximizing the distance between them.

    Args:
        batch_size: Number of samples to select.
        metric: Metric to use for distance computation.
    """
    def __init__(self, batch_size: int, metric: str = 'euclidean'):
        super().__init__(batch_size)
        self.metric = metric

    def fit(self, X: np.array, y: np.array = None) -> 'RankedBatchSampler':
        """Does nothing, RankedBatch is unsupervised.

        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
       
        Returns:
            The object itself
        """
        self.X_selected = X
        return self

    def select_samples(self, X: np.array,
                       samples_weights: np.array) -> np.array:
        """Selects the samples to annotate from unlabelled data.
        
        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).
            sample_weights: Weights of the samples of shape (n_samples).

        Returns:
            Indices of the selected samples of shape (batch_size).
        """
        if self._not_enough_samples(X):
            return np.arange(X.shape[0])

        n_unlabeled = X.shape[0]
        n_samples = n_unlabeled + self.X_selected.shape[0]

        # We are going to modify this array so we copy it
        samples_weights = samples_weights.copy()

        # We compute the distances for labeled data in 2 steps
        _, similarity_scores = pairwise_distances_argmin_min(
            X, self.X_selected, metric=self.metric)
        similarity_scores = 1 / (1 + similarity_scores)

        selected_samples = []

        for _ in range(self.batch_size):
            alpha = n_unlabeled / n_samples
            scores = (alpha * (1 - similarity_scores)
                      + (1 - alpha) * samples_weights)

            idx_furthest = np.argmax(scores)
            selected_samples.append(idx_furthest)

            # Update similarities considering the selected sample as labeled
            # We could remove its value from the array but we avoid realloc
            sim = 1 / (1 + pairwise_distances(
                X, X[idx_furthest, None],
                metric=self.metric)[:, 0])
            similarity_scores = np.max([similarity_scores, sim], axis=0)
            samples_weights[idx_furthest] = 0.
            n_unlabeled -= 1

        return np.asarray(selected_samples)
