from .version import check_modules

check_modules('submodular', 'submodularity')  # noqa

import numpy as np
from apricot import FacilityLocationSelection
from sklearn.metrics import pairwise_distances

from .base import BaseQuerySampler
from .typeutils import MetricType


class SubmodularSampler(BaseQuerySampler):
    """Select samples using a facility location selector
    
    Args:
        batch_size: Number of samples to select.
        metric: Metric to use for distance computation.
        n_jobs: Number of jobs to run in parallel. -1 means using all cores.
    """

    def __init__(self, batch_size: int, metric: MetricType = 'euclidean',
                 n_jobs: int = 1):
        super().__init__(batch_size)
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X: np.array, y: np.array = None) -> 'SubmodularSampler':
        """Does nothing.
        
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        return self

    def select_samples(self, X: np.array) -> np.array:
        """Select the best samples using submodular optimization.

        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).

        Returns:
            Indices of the selected samples of shape (batch_size).
        """
        if self._not_enough_samples(X):
            return np.arange(X.shape[0])

        if self.metric != 'precomputed':
            distances = pairwise_distances(
                X, metric=self.metric, n_jobs=self.n_jobs)
            model = FacilityLocationSelection(
                self.batch_size, pairwise_func='precomputed').fit(distances)
        else:  
            model = FacilityLocationSelection(
                self.batch_size, pairwise_func='precomputed').fit(X)
        return model.ranking
