from .base import BaseQuerySampler
import numpy as np
import pandas as pd
from libact.base.dataset import Dataset
from libact.models import SklearnProbaAdapter


class LibactWrapper(BaseQuerySampler):

    def __init__(self, estimator, libact_class, batch_size, init_params={}):
        super().__init__()
        self.estimator = estimator
        self.libact_class = libact_class
        self.batch_size = batch_size
        self.init_params = init_params

    def fit(self, X, y):

        # Libact object requires to have access to the whole data at init time.
        # UGLY We store it, no other choice.
        self.X_train = X
        self.y_train = y

    def predict(self, X):

        # Libact do its selection on the whole dataset. Merging train and test:
        n_train, n_test = self.X_train.shape[0], X.shape[0]
        X = np.vstack([self.X_train, X])
        # y is of shape n_train + n_test, we fill the test samples with None for each class
        y = self.y_train.tolist() + [None] * n_test


        model = self.libact_class(
            model=SklearnProbaAdapter(self.estimator),
            dataset=Dataset(X, y),
            **self.init_params)

        _, selected_idx = model.make_query(return_score=True)
        selected_idx = [i - n_train for i, _ in sorted(selected_idx, key = lambda x: x[1])[-self.batch_size:]]
        selected = np.zeros(n_test, dtype=int)
        selected[selected_idx] = 1

        return selected
