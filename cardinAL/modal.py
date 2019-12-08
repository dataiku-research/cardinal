from .base import BaseQuerySampler
import numpy as np


class ModalWrapper(BaseQuerySampler):

    def __init__(self, learner, batch_size, refit=False):
        super().__init__()
        self.learner = learner
        self.batch_size = batch_size
        self.refit = refit

    def fit(self, X, y):

        if self.refit:
            self.learner.fit(X, y)
        else:
            self.learner._add_training_data(X, y)

        return self

    def predict(self, X):

        selected_idx, _ = self.learner.query(X, n_instances=self.batch_size)

        selected = np.zeros(X.shape[0], dtype=int)
        selected[selected_idx] = 1

        return selected
