import numpy as np
from .base import BaseQuerySampler
from sklearn.utils import check_random_state


class RandomSampler(BaseQuerySampler):
    """Randomly select samples

    Parameters
    ----------
    batch_size : float, default: 0.05
        If batch_size < 1., it is interpreted as fraction of the training
        set to select for labeling. If batch_size >= 1., it is interpreted
        as the number of samples to draw.
    verbose : integer, optional
        The verbosity level
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    Attributes
    ----------
    pipeline_ : sklearn.pipeline
        Pipeline used to predict the class probability.
    """

    def __init__(self, batch_size=0.05, shuffle=True,
                 random_state=None):
        super().__init__(batch_size=batch_size)
        self.shuffle = shuffle
        self.random_state = random_state

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
        self.random_state = check_random_state(self.random_state)
        return self

    def score_samples(self, X):
        return self.random_state.rand(X.shape[0])
