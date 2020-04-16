import numpy as np


class ContradictionMetric():

    def __init__(self):
        self.reset()

    def accumulate(self, n_samples, probas_test):
        if self.last_probas_test is not None:
            self.contradictions.append(np.abs(probas_test - self.last_probas_test).sum())
            self.n_samples.append(n_samples)
        self.last_probas_test = probas_test

    def get_x_y(self):
        return (self.n_samples, self.contradictions)

    def reset(self):
        self.contradictions = []
        self.n_samples = []
        self.last_probas_test = None
