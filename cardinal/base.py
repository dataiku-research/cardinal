from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin


class BaseQuerySampler(ClusterMixin, TransformerMixin, BaseEstimator):
    """Base class providing utils for query samplers
    A query sampler can be seen as a clustering since it is most of the time
    an unsupervised approach sorting out samples to be annotated and those who
    should not.
    This is also considered a transformer for the sole purpose of chaining them.
    It is common in sampling to chain several approached. However, this can be
    easily dropped since it is more of a hack than a feature.
    """

    def __init__(self):
        self.labels_ = None

    def predict(self, X):
        raise NotImplementedError

    def transform(self, X):
        self.labels_ = self.predict(X).astype(bool)
        return X[self.labels_]

    def inverse_transform(self, X):
        if self.labels_ is None:
            return X
        unmask = self.labels_.copy()
        unmask[unmask == 1] = X
        return unmask


class ChainQuerySampler(BaseQuerySampler):
    """Allows to whain query sampling methods
    This strategy is usually used to chain a simple query sampler with a
    more complex one. The first query sampler is used to reduce the
    dimensionality.
    """

    def __init__(self, *sampler_list):
        self.sampler_list = sampler_list

    def fit(self, X, y=None):
        # Fits only the first one. The other will depend on this one.
        self.sampler_list[0].fit(X, y)
    
    def predict(self, X):
        selected = self.sampler_list[0].predict(X).astype(bool)

        for sampler in self.sampler_list[1:]:
            new_selected = sampler.fit_predict(X[selected])
            selected[selected] = new_selected
        
        return selected