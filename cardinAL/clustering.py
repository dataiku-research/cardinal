import numpy as np
from sklearn.cluster import KMeans

from .base import BaseQuerySampler
from .uncertainty import ConfidenceSampler


class KCentroidSampler(BaseQuerySampler):
    """ KCentroid based query sampler.
    In order to increase diversity, it is possible to use a centroid based
    clustering to select samples.

    Parameters
    ----------
    clustering : sklearn estimator
        A clustering algorithm that must feature a transform method that
        returns the distance of samples from centroids.
    batch_size : int
        Number of samples to draw when predicting.
    verbose : integer, optional
        The verbosity level
    Attributes
    ----------
    clustering_ : sklearn estimator
        The fitted clustering estimator.
    """
    def __init__(self, clustering, batch_size, verbose=0):
        super().__init__(batch_size)
        self.clustering_ = clustering
        self.verbose = verbose

    def fit(self, X, y=None):
        """Does nothing.
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
        return self

    def select_samples(self, X, sample_weight=None):
        """Fits clustering on the samples and select the ones closest to centroids.
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
        model = self.clustering_.fit(X, sample_weight=sample_weight)
        closest = np.argmin(model.transform(X), axis=0)
        return closest


class KMeansSampler(KCentroidSampler):
    """Query sampler that uses a KMeans approach to increase selection diversity.

    Parameters
    ----------
    batch_size : int
        Number of samples to draw when predicting.
    verbose : integer, optional
        The verbosity level
    Attributes
    ----------
    clustering_ : sklearn estimator
        The fitted clustering estimator.
    """

    def __init__(self, batch_size, verbose=0, **kmeans_args):
        if 'n_clusters' in kmeans_args:
            raise ValueError(
                'You have specified n_clusters={} when creating KMeansSampler.'
                ' This is not supported since n_clusters is overridden using '
                'batch_size.'.format(kmeans_args['n_clusters'])
        kmeans_args['n_clusters'] = batch_size
        super().__init__(KMeans(**kmeans_args), batch_size, verbose)