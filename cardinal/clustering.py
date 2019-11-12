from .base import BaseQuerySampler
import numpy as np
from sklearn.cluster import KMeans
from .uncertainty import UncertaintySampler


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
        super().__init__()
        self.clustering_ = clustering
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
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
        self._classes = [0, 1]  
        return self

    def predict(self, X, sample_weight=None):
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
        selected_samples = np.zeros(X.shape[0])
        selected_samples[closest] = 1
        return selected_samples


class KMeansSampler(KCentroidSampler):
    """Query sampler that uses a KMeans approach to increase selection diversity.

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

    def __init__(self, clustering, batch_size, verbose=0, **kmeans_args):
        super().__init__(KMeans(**kmeans_args), batch_size, verbose)


class WKMeansSampler(BaseQuerySampler):

    def __init__(self, pipeline, beta=10, batch_size=0.05, shuffle=True,
                 verbose=0, random_state=None,
                 n_iter_no_change=5, class_weight=None):
        super().__init__()

        self.uncertainty = UncertaintySampler(
            pipeline,
            beta * batch_size,
            shuffle,
            verbose,
            random_state,
            n_iter_no_change,
            class_weight)

        self.kmeans = KMeansSampler(
            batch_size,
            shuffle,
            verbose,
            random_state,
            n_iter_no_change,
            class_weight
        )

    def fit(self, X, y):
        self.uncertainty.fit(X, y)

    def predict(self, X):
        selected = self.uncertainty.predict(X).astype(bool)
        X_selected = X[selected]
        k_selected = self.kmeans.predict(X_selected)
        selected[selected] = k_selected
        return selected.astype(int)