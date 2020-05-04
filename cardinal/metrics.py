import numpy as np


class ContradictionMetric():
    """Stores the amount of contradictions along an experiment

    We call contradiction the difference between predictions of two successive
    models on an isolated test set.
    """
    def __init__(self):
        self.reset()

    """Stores contradiction for a new iteration.

    Args:
        n_samples : Number of training samples
        probas_test : Predictions of shape (n_samples, n_classes)
    """
    def accumulate(self, n_samples: int, probas_test: np.array):
        if self.last_probas_test is not None:
            self.contradictions.append(
                np.abs(probas_test - self.last_probas_test).sum())
            self.n_samples.append(n_samples)
        self.last_probas_test = probas_test

    """Returns the recorded metrics
    """
    def get_x_y(self):
        return (self.n_samples, self.contradictions)

    """Reset the metrics for a new experiment
    """
    def reset(self):
        self.contradictions = []
        self.n_samples = []
        self.last_probas_test = None
