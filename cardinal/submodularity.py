from apricot import FacilityLocationSelection
from .base import BaseQuerySampler
import numpy as np
from sklearn.metrics import pairwise_distances


class SubmodularSampler(BaseQuerySampler):
    """TODO
    Documentation.

    Parameters
    ----------
    batch_size : float, default: 0.05
        If batch_size < 1., it is interpreted as fraction of the training
        set to select for labeling. If batch_size >= 1., it is interpreted
        as the number of samples to draw.
    verbose : integer, optional
        The verbosity level

    Attributes
    ----------
    pipeline_ : sklearn.pipeline
        Pipeline used to predict the class probability.
    """

    def __init__(self, batch_size, compute_distances=False, verbose=0):
        super().__init__()
        # TODO: can we check that the pipeline has a predict_proba?
        self.batch_size = batch_size
        self.compute_distances = compute_distances
        self.verbose = verbose

    def fit(self, X, y):
        """Fit linear model with Stochastic Gradient Descent.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data
        y : numpy array, shape (n_samples,)
            Target values
        Returns
        -------
        self : returns an instance of self.
        """
        self._classes = [0, 1]  
        return self

    def predict(self, X):
        if self.compute_distances:
            model = FacilityLocationSelection(self.batch_size, pairwise_func='precomputed').fit(pairwise_distances(X))
        else:  
            model = FacilityLocationSelection(self.batch_size).fit(X)
        selected_samples = np.zeros(X.shape[0])
        selected_samples[model.ranking] = 1
        return selected_samples