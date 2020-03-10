# Note: This code is inspired from modAL implementation
# https://modal-python.readthedocs.io/en/latest/content/query_strategies/ranked_batch_mode.html

import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

from .base import BaseQuerySampler


class RankedBatchSampler(BaseQuerySampler):
    """TODO

    Parameters
    ----------
    batch_size : int
        Number of samples to draw when predicting.
    verbose : integer, optional
        The verbosity level
    Attributes
    ----------
    pipeline_ : sklearn.pipeline
        Pipeline used to predict the class probability.
    """

    def __init__(self, batch_size, verbose=0):
        super().__init__(batch_size)
        self.verbose = verbose

    def fit(self, X, y=None):
        """Does nothing, all data must be passed at sample selection.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data
        Returns
        -------
        self : returns an instance of self.
        """
        return self

    def select_samples(self, X, samples_weights):
        """Selects the samples to annotate from unlabelled data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data
        sample_weights : numpy array, shape (n_samples,)
            Weights of the samples. Set labeled samples as -1.
        Returns
        -------
        self : returns an instance of self.
        """

        n_samples = X.shape[0]
        index = np.arange(n_samples)
        unlabeled_mask = (samples_weights > .5)
        n_unlabeled = unlabeled_mask.sum()

        # We are going to modify this array so we copy it
        samples_weights = samples_weights.copy()

        # We compute the distances for labeled data in 2 steps
        # TODO: can be parallelized
        _, similarity_scores = pairwise_distances_argmin_min(
            X[unlabeled_mask], X[np.logical_not(unlabeled_mask)], metric='euclidean')
        similarity_scores = 1 / (1 + similarity_scores)

        selected_samples = []

        for _ in range(self.batch_size):

            alpha = n_unlabeled / n_samples
            scores = alpha * (1 - similarity_scores) + (1 - alpha) * samples_weights[unlabeled_mask]

            idx_furthest = index[unlabeled_mask][np.argmax(scores)]
            selected_samples.append(idx_furthest)

            # Update the distances considering this sample as reference one
            distances_to_furthest = pairwise_distances(X[unlabeled_mask], X[idx_furthest, None], metric='euclidean')[:, 0]
            similarity_scores = np.max([similarity_scores, 1 / (1 + distances_to_furthest)], axis=0)
            samples_weights[idx_furthest] = 0.
            n_unlabeled -= 1

        return selected_samples
