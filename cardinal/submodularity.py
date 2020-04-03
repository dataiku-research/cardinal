from .version import check_modules

check_modules('submodular', 'submodularity')  # noqa

from apricot import FacilityLocationSelection
from sklearn.metrics import pairwise_distances

from .base import BaseQuerySampler


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
        super().__init__(batch_size)
        # TODO: can we check that the pipeline has a predict_proba?
        self.compute_distances = compute_distances
        self.verbose = verbose

    def fit(self, X, y=None):
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

    def select_samples(self, X):
        if self.compute_distances:
            model = FacilityLocationSelection(self.batch_size, pairwise_func='precomputed').fit(pairwise_distances(X))
        else:  
            model = FacilityLocationSelection(self.batch_size).fit(X)
        return model.ranking
