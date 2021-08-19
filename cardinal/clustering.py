import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

from .base import BaseQuerySampler
from .version import check_modules
from .kmeans import IncrementalMiniBatchKMeans
from .uncertainty import MarginSampler


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
        self.fixed_cluster_centers = None


    def fit(self, X, y=None) -> 'KCentroidSampler':
        """Does nothing, this method is unsupervised.
        
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        self.fixed_cluster_centers = X
        return self


    def select_samples(self, X: np.array,
                       sample_weight: np.array = None,
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
        if self.fixed_cluster_centers is not None:
            kwargs['fixed_cluster_centers'] = self.fixed_cluster_centers
            n_fixed_clusters = self.fixed_cluster_centers.shape[0]
        if recenter_every is not None:
            kwargs['recenter_every'] = recenter_every
        self.clustering_.n_clusters = self.batch_size + n_fixed_clusters
        model = self.clustering_.fit(X, **kwargs)
        distances = model.transform(X)[:, n_fixed_clusters:]

        # Sometimes, one sample can be the closest to two centroids. In that
        # case, we want to take the second closest one, etc.
        # linear_sum_assignemnt solves this problem.
        return linear_sum_assignment(distances)[0]


class TwoStepKCentroidSampler(BaseQuerySampler):
    """KMeans sampler using a margin uncertainty sampler as preselector

    """

    def __init__(self, kcentroid_sampler, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):
        
        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            kcentroid_sampler(batch_size, **kmeans_args)
        ]

    def fit(self, X: np.array, y: np.array = None) -> 'TwoStepKMeansSampler':
        """Fits the first query sampler

        Args:
            X: Labeled samples of shape [n_samples, n_features].
            y: Labels of shape [n_samples].
        
        Returns:
            The object itself
        """
        self.sampler_list[0].fit(X, y)
        return self

    def select_samples(self, X: np.array,
                       sample_weight: np.array = None) -> np.array:
        """Selects the using uncertainty preselection and KMeans sampler.

        Args:
            X: Pool of unlabeled samples of shape (n_samples, n_features).
            sample_weight: Weight of the samples of shape (n_samples),
                optional.

        Returns:
            Indices of the selected samples of shape (batch_size).
        """
        selected = self.sampler_list[0].select_samples(X)
        kwargs = dict()
        if sample_weight is not None:
            kwargs['sample_weight'] = sample_weight[selected]
        new_selected = self.sampler_list[1].select_samples(
            X[selected], **kwargs)
        selected = selected[new_selected]
        
        return selected

class TwoStepIWKMeansSampler(TwoStepKCentroidSampler):

    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):
        
        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            IncrementalMiniBatchKMeansSampler(batch_size, **kmeans_args)
        ]


class KCenterGreedy(BaseQuerySampler):
    """ KCenter greedy query sampler.
    Select the furthest sample from already select ones, add it to the
    selected, and repeat until batch_size is reached.

    Args:
        batch_size: Number of samples to draw when predicting.
    """
    def __init__(self, embedding_fun, batch_size, metric='euclidean'):
        super().__init__(batch_size)
        self._embedding_fun = embedding_fun
        self.metric = metric

    def fit(self, X, y=None) -> 'KCenterGreedy':
        """Does nothing, this method is unsupervised.
        
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        self._X_centers = self._embedding_fun(X)
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

        selected = []
        X = self._embedding_fun(X)

        _, distances = pairwise_distances_argmin_min(X, self._X_centers, metric=self.metric)


        for _ in range(self.batch_size):

            # Select the point furthest from already selected
            selected.append(np.argmax(distances))

            # Consider this point added to label by updating distances
            distances_to_new = pairwise_distances(X, X[selected[-1], None], metric=self.metric)[:, 0]
            distances = np.min([distances, distances_to_new], axis=0)

        # Return numpy array, not a list.
        return np.asarray(selected)
