import logging
from abc import ABC, abstractmethod

import numpy as np


class BaseMonitor(ABC):
    """A monitor is a metric and a set of utils to record it and monitor it.

    Args:
        batch_size: If specified, a warning will be issued if batch_size is not correct
        tolerance: 

    """

    def __init__(self, batch_size=None, tolerance=None):
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.reset()

    def reset(self):
        self.n_samples = []
        self.values = []

    def _append_n_samples(self, n_samples):
        self.n_samples.append(n_samples)
        if not self.batch_size or len(self.n_samples) <= 1:
            return
        this_batch_size = self.n_samples[-1] - self.n_samples[-2]
        if this_batch_size != self.batch_size:
            logging.warn(
                'Batch size of iteration {} is {} which is different'
                'from the reference batch size {}'.format(
                    len(self.n_samples), this_batch_size, self.batch_size
                )
            )

    @abstractmethod
    def accumulate(self, n_samples, value):
        pass

    @abstractmethod
    def get(self):
        pass

    def is_stalled(self, n_iter=1):
        if len(self.values) < n_iter + 1:
            return False
        for prev_v, curr_v in zip(self.values[-n_iter - 1:-1], self.values[-n_iter]):
            if np.abs(curr_v - prev_v) > self.tolerance:
                return False
        return True
        

class ContradictionMonitor(BaseMonitor):
    """Stores the amount of contradictions along an experiment

    We call contradiction the difference between predictions of two successive
    models on an isolated test set.
    """

    """Stores contradiction for a new iteration.

    Args:
        n_samples : Number of training samples
        probas_test : Predictions of shape (n_samples, n_classes)
    """
    def accumulate(self, n_samples: int, probas_test: np.array):
        if self.last_probas_test is not None:
            self.values.append(
                np.abs(probas_test - self.last_probas_test).sum())
            self._append_n_samples(n_samples)
        self.last_probas_test = probas_test

    """Returns the recorded metrics
    """
    def get(self):
        return {
            "n_samples": self.n_samples,
            "contradictions": self.values
        }

    """Reset the metrics for a new experiment
    """
    def reset(self):
        super().reset()
        self.last_probas_test = None
