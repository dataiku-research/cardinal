import numpy as np

from sklearn.base import BaseEstimator


class BaseQuerySampler(BaseEstimator):
    """Base interface for query samplers
    
    A query sampler is an object that takes as input labeled and/or unlabled samples
    and use knowledge from them to selected the most informative ones.
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def fit(self, X, y=None):
        pass

    def select_samples(self, X):
        """Selects the samples to annotate from unlabeled data.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Samples to evaluate.

        Returns:
            predictions (np.array): Indices of the selected samples.
        """
        raise NotImplementedError


class ScoredQuerySampler(BaseEstimator):
    """Base class providing utils for query samplers
    A query sampler can be seen as a clustering since it is most of the time
    an unsupervised approach sorting out samples to be annotated and those who
    should not.
    This is also considered a transformer for the sole purpose of chaining them.
    It is common in sampling to chain several approached. However, this can be
    easily dropped since it is more of a hack than a feature.
    """

    def __init__(self, batch_size, strategy='top'):
        self.batch_size = batch_size
        self.strategy = strategy
        pass

    def fit(self, X, y=None):
        pass

    def score_samples(self, X):
        raise NotImplementedError

    def select_samples(self, X):
        """Selects the samples to annotate from unlabelled data using the internal scoring.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Samples to evaluate.
            strategy (str): Strategy to use to select queries. Can top, linear_choice, or squared_choice.

        Returns:
            predictions (np.array): Indices of the selected samples.
        """
        sample_scores = self.score_samples(X)
        self.sample_scores_ = sample_scores
        if self.strategy == 'top':
            index = np.argsort(sample_scores)[-self.batch_size:]
        elif self.strategy == 'linear_choice':
            index = np.random.choice(
                np.arange(X.shape[0]), k=self.batch_size,
                replace=False, p=sample_scores / np.sum(sample_scores))
        else:
            raise ValueError('Unknown sample selection stretegy {}'.format(self.strategy))
        return index


class ChainQuerySampler(BaseQuerySampler):
    """Allows to chain query sampling methods
    This strategy is usually used to chain a simple query sampler with a
    more complex one. The first query sampler is used to reduce the
    dimensionality.
    """

    def __init__(self, *sampler_list):
        self.sampler_list = sampler_list

    def fit(self, X, y=None):
        # Fits only the first one. The other will depend on this one.
        self.sampler_list[0].fit(X, y)
    
    def select_samples(self, X):
        selected = self.sampler_list[0].select_samples(X)

        for sampler in self.sampler_list[1:]:
            sampler.fit(X)
            new_selected = sampler.predict(X[selected])
            selected = selected[new_selected]
        
        return selected
