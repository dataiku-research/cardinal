import numpy as np

from .version import check_modules

check_modules('sklearn', 'zhdanov2019')  # noqa

from .base import BaseQuerySampler
from .uncertainty import MarginSampler
from .clustering import KMeansSampler


class TwoStepKMeansSampler(BaseQuerySampler):
    """KMeans sampler using a margin uncertainty sampler as preselector

    """

    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):
        
        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            KMeansSampler(batch_size, **kmeans_args)
        ]

    def fit(self, X: np.array, y: np.array = None) -> 'TwoStepKMeansSampler':
        """Fits the first query sampler

        Args:
            X: Labeled samples of shape [n_samples, n_features].
            y: Labels of shape [n_samples].
        
        Returns:
            The object itself
        """
        for sampler in self.sampler_list:
            sampler.fit(X, y)
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
