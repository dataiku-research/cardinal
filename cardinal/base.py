from typing import List
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np

from .typeutils import (RandomStateType, check_random_state,
                        NotEnoughSamplesWarning)


class BaseQuerySampler(ABC):
    """Abstract Base Class for query samplers
    
    A query sampler is an object that takes as input labeled and/or unlabeled
    samples and use knowledge from them to selected the most informative ones.

    Args:
        batch_size: Numbers of samples to select.
    """
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit the model on labeled samples.

        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        pass

    @abstractmethod
    def select_samples(self, X: np.array) -> np.array:
        """Selects the samples to annotate from unlabeled data.

        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).

        Returns:
            Indices of the selected samples of shape (batch_size).
        """
        pass

    def _not_enough_samples(self, X: np.array) -> bool:
        cond = X.shape[0] < self.batch_size
        if cond:
            warn(f'''Requested {self.batch_size} samples but data only
             has {X.shape[0]}. All available data will be returned''',
                 NotEnoughSamplesWarning)
        return cond


class ScoredQuerySampler(BaseQuerySampler):
    """Abstract Base Class handling query samplers relying on a total order.
    Query sampling methods often scores all the samples and then pick samples
    using these scores. This base class handles the selection system, only
    a scoring method is then required.

    Args:
        batch_size: Numbers of samples to select.
        strategy: Describes how to select the samples based on scores. Can be
                  "top", "weighted".
        random_state: Random seeding
    """
    def __init__(self, batch_size: int, strategy: str = 'top',
                 random_state: RandomStateType = None):
        super().__init__(batch_size)
        self.strategy = strategy
        self.random_state = check_random_state(random_state)

    @abstractmethod
    def score_samples(self, X: np.array) -> np.array:
        """Give an informativeness score to unlabeled samples.

        Args:
            X: Samples to evaluate.

        Returns:
            Scores of the samples.
        """
        pass

    def select_samples(self, X: np.array) -> np.array:
        """Selects the samples from unlabeled data using the internal scoring.

        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).
            strategy: Strategy to use to select queries. Can be one of top,
                      linear_choice, or squared_choice.

        Returns:
            Indices of the selected samples of shape (batch_size).
        """
        if self._not_enough_samples(X):
            return np.arange(X.shape[0])

        sample_scores = self.score_samples(X)
        self.sample_scores_ = sample_scores
        if self.strategy == 'top':
            index = np.argsort(sample_scores)[-self.batch_size:]
        elif self.strategy == 'weighted':
            index = self.random_state.choice(
                np.arange(X.shape[0]), size=self.batch_size,
                replace=False, p=sample_scores / np.sum(sample_scores))
        else:
            raise ValueError('Unknown sample selection strategy {}'
                             .format(self.strategy))
        return index
