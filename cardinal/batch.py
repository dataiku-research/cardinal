# Note: This code is inspired from modAL implementation
# https://modal-python.readthedocs.io/en/latest/content/query_strategies/ranked_batch_mode.html

from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from scipy.stats import entropy
import numpy as np
from .base import BaseQuerySampler
from sklearn.metrics import pairwise_distances


class RankedBatchSampler(BaseQuerySampler):
    """TODO

    Parameters
    ----------
    query_sampler : cardinAL.BaseQuerySampler
        A query sampler which scores will be used to score the batch of samples
    alpha: float, optional
        Weight of the sample similarity in the final score.
    TODO This is a duplicate of the property in the query_sampler
    batch_size : int
        Number of samples to draw when predicting.
    verbose : integer, optional
        The verbosity level
    Attributes
    ----------
    pipeline_ : sklearn.pipeline
        Pipeline used to predict the class probability.
    """

    def __init__(self, query_sampler, alpha, batch_size, verbose=0):
        super().__init__()
        self.query_sampler = query_sampler
        self.alpha = alpha
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the estimator on labeled samples.
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
        
        # We delegate pretty much everything to the estimator
        self.query_sampler.fit(X, y)

        # UGLY This is something we would like to avoid but as of now, not possible
        # Note: This approach is a bit similar yet more exhaistive than the K_means one
        # An in-between could be to compute a KMean on trained data and use it as a
        # "repulsive" force on unlabeled data. We could also use random projections to
        # make the data "smaller"

        self.X_train = X
        
        return self

    def predict(self, X):
        """Selects the samples to annotate from unlabelled data.
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

        if self.alpha == 'auto':
            alpha = X.shape[0] / (X.shape[0] + self.X_train.shape[0])
        else:
            alpha = self.alpha

        # UGLY This is done to get the prediction scores.
        # There are several options to avoid this:
        # - Use predit_proba but the score returned is not really a proba
        # - Have a more generic object to make this code easier without copy pasting everything

        self.query_sampler.predict(X)
        uncertainty = self.query_sampler.confidence_

        # We compute the distances for labeled data
        # TODO: can be parallelized
        distances = pairwise_distances(X, self.X_train, metric='euclidean').min(axis=1)

        selected_samples = np.zeros(X.shape[0])

        for _ in range(self.batch_size):

            similarity_scores = 1 / (1 + distances)
            scores = alpha * (1 - similarity_scores) + (1 - alpha) * uncertainty

            idx_furthest = np.argmax(scores)
            selected_samples[idx_furthest] = True

            # Update the distances considering this sample as reference one
            distances_to_furthest = pairwise_distances(X, X[idx_furthest], metric='euclidean')
            distances = np.min([distances, distances_to_furthest], axis=0)

        return X[selected_samples]


