import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import BaseQuerySampler
from .version import check_modules
from .kmeans import IncrementalMiniBatchKMeans


class KCentroidSampler(BaseQuerySampler):
    """ KCentroid based query sampler.
    In order to increase diversity, it is possible to use a centroid based
    clustering to select samples.

    Args:
        clustering: A clustering algorithm matching the sklearn interface
        batch_size: Number of samples to draw when predicting.

    Attributes:
        clustering_ : The fitted clustering estimator.
    """
    def __init__(self, clustering, batch_size):
        super().__init__(batch_size)
        self.clustering_ = clustering

    def fit(self, X, y=None) -> 'KCentroidSampler':
        """Does nothing, this method is unsupervised.
        
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        return self

    def select_samples(self, X: np.array,
                       sample_weight: np.array = None) -> np.array:
        """Clusters the samples and select the ones closest to centroids.
        
        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).
            sample_weight: Weight of the samples of shape (n_samples),
                optional.

        Returns:
            Indices of the selected samples of shape (batch_size).
        """
        if self._not_enough_samples(X):
            return np.arange(X.shape[0])

        kwargs = dict(sample_weight=sample_weight) if (sample_weight is not None) else dict()
        model = self.clustering_.fit(X, **kwargs)
        distances = model.transform(X)

        # Sometimes, one sample can be the closest to two centroids. In that
        # case, we want to take the second closest one, etc.
        # linear_sum_assignemnt solves this problem.
        return linear_sum_assignment(distances)[0]


class KMeansSampler(KCentroidSampler):
    """Select samples as closest sample to KMeans centroids.

    Args:
        batch_size: Number of samples to draw when predicting.
    """
    def __init__(self, batch_size, **kmeans_args):
        check_modules('sklearn', 'clustering.KmeansSampler')
        from sklearn.cluster import KMeans

        if 'n_clusters' in kmeans_args:
            raise ValueError(
                'You have specified n_clusters={} when creating KMeansSampler.'
                ' This is not supported since n_clusters is overridden using '
                'batch_size.'.format(kmeans_args['n_clusters']))
        kmeans_args['n_clusters'] = batch_size
        super().__init__(KMeans(**kmeans_args), batch_size)


class MiniBatchKMeansSampler(KCentroidSampler):
    """Select samples as closest sample to MiniBatchKMeans centroids.

    Args:
        batch_size: Number of samples to draw when predicting.
    """
    def __init__(self, batch_size, **kmeans_args):
        check_modules('sklearn', 'clustering.MiniBatchKmeansSampler')
        from sklearn.cluster import MiniBatchKMeans

        if 'n_clusters' in kmeans_args:
            raise ValueError(
                'You have specified n_clusters={} when creating '
                'MiniBatchKMeansSampler. This is not supported since'
                'n_clusters is overridden using '
                'batch_size.'.format(kmeans_args['n_clusters']))
        kmeans_args['n_clusters'] = batch_size
        super().__init__(MiniBatchKMeans(**kmeans_args), batch_size)

class IncrementalMiniBatchKMeansSampler(KCentroidSampler):
    """Select samples as closest sample to MiniBatchKMeans centroids.

Args:
        batch_size: Number of samples to draw when predicting.
    """
    def __init__(self, batch_size, **kmeans_args):
        if 'n_clusters' in kmeans_args:
            raise ValueError(
                'You have specified n_clusters={} when creating '
                'MiniBatchKMeansSampler. This is not supported since'
                'n_clusters is overridden using '
                'batch_size.'.format(kmeans_args['n_clusters']))
        kmeans_args['n_clusters'] = batch_size
        super().__init__(IncrementalMiniBatchKMeans(**kmeans_args), batch_size)

    def select_samples(self, X: np.array,
                       sample_weight: np.array = None,
                       fixed_cluster_centers: np.array=None,
                       recenter_every=None) -> np.array:
        """Clusters the samples and select the ones closest to centroids.
        
        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).
            sample_weight: Weight of the samples of shape (n_samples),
                optional.

        Returns:
            Indices of the selected samples of shape (batch_size).
        """
        if self._not_enough_samples(X):
            return np.arange(X.shape[0])

        kwargs = dict()
        n_fixed_clusters = 0
        if sample_weight is not None:
            kwargs['sample_weight'] = sample_weight
        if fixed_cluster_centers is not None:
            kwargs['fixed_cluster_centers'] = fixed_cluster_centers
            n_fixed_clusters = fixed_cluster_centers.shape[0]
        if recenter_every is not None:
            kwargs['recenter_every'] = recenter_every
        self.clustering_.n_clusters = self.batch_size + n_fixed_clusters
        model = self.clustering_.fit(X, **kwargs)
        distances = model.transform(X)[:, n_fixed_clusters:]

        # Sometimes, one sample can be the closest to two centroids. In that
        # case, we want to take the second closest one, etc.
        # linear_sum_assignemnt solves this problem.
        return linear_sum_assignment(distances)[0]
