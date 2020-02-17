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

    def __init__(self, query_sampler, batch_size, verbose=0):
        super().__init__(batch_size)
        self.query_sampler = query_sampler
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
        # We delegate pretty much everything to the estimator
        self.query_sampler.fit(X, y)

        # UGLY This is something we would like to avoid but as of now, not possible
        # Note: This approach is a bit similar yet more exhaistive than the K_means one
        # An in-between could be to compute a KMean on trained data and use it as a
        # "repulsive" force on unlabeled data. We could also use random projections to
        # make the data "smaller"

        self.X_train = X
        
        return self

    def select_samples(self, X):
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

        n_unlabeled = X.shape[0]
        n_labeled = self.X_train.shape[0]

        # UGLY This is done to get the prediction scores.
        # There are several options to avoid this:
        # - Use predit_proba but the score returned is not really a proba
        # - Have a more generic object to make this code easier without copy pasting everything

        uncertainty = self.query_sampler.score_samples(X)

        # We compute the distances for labeled data
        # TODO: can be parallelized
        similarity_scores = 1 / (1 + pairwise_distances(X, self.X_train, metric='euclidean').min(axis=1))

        selected_samples = []

        for _ in range(self.batch_size):

            alpha = n_unlabeled / (n_unlabeled + n_labeled)
            scores = alpha * (1 - similarity_scores) + (1 - alpha) * uncertainty

            idx_furthest = np.argmax(scores)
            selected_samples.append(idx_furthest)

            # Update the distances considering this sample as reference one
            distances_to_furthest = pairwise_distances(X, X[idx_furthest, None], metric='euclidean')[:, 0]
            similarity_scores = np.max([similarity_scores, 1 / (1 + distances_to_furthest)], axis=0)

            n_labeled += 1
            n_unlabeled -= 1

        return selected_samples


