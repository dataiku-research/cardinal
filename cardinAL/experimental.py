import numpy as np
from .base import BaseQuerySampler
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from .uncertainty import _get_probability_classes, confidence_sampling
from .clustering import KMeansSampler


class DeltaSampler(BaseQuerySampler):
    """Look at samples for which the last predictions are the same.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        Pipeline used to determine the prediction confidence. For this method
        it must be a classifier with a predict_proba method.
    batch_size : int
        Number of samples to draw when predicting.
    verbose : integer, optional
        The verbosity level
    Attributes
    ----------
    pipeline_ : sklearn.pipeline
        Pipeline used to predict the class probability.
    """

    def __init__(self, pipeline, batch_size, verbose=0, n_last=1):
        super().__init__()
        # TODO: can we check that the pipeline has a predict_proba?
        self.pipeline_ = pipeline
        self._previous_pipeline = []
        self._current_pipeline = None
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_last = n_last

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
        
        if self._current_pipeline is not None:
            self._previous_pipeline.append(self._current_pipeline)

        # We delegate pretty much everything to the estimator
        self.pipeline_.fit(X, y)
        self._current_pipeline = clone(self.pipeline_).fit(X, y)
        
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
        selected_samples = np.zeros(X.shape[0])

        if len(self._previous_pipeline) > 0:
            try:
                confidence = np.asarray([model.predict_proba(X) for model in self._previous_pipeline[-self.n_last:] + [self._current_pipeline]])
                confidence = np.abs(confidence[:-1] - confidence[1:]).sum(axis=2).sum(axis=0)
            except ValueError:
                print('In case of unbalanced classes, shapes may not be compatible')
                confidence = np.random.random(X.shape[0])
        else:
            print('not fitted, doing random') # TODO better strategy when not fitted
            confidence = np.random.random(X.shape[0])

        # TODO use np.argpart instead of argsort for newer versions of numpy
        index = np.flip(np.argsort(confidence))[:self.batch_size]
        
        self.confidence_ = confidence
        
        selected_samples[index] = 1
        self.labels_ = selected_samples

        return selected_samples


class ProbaKMeans(BaseQuerySampler):
    """Does stuff
    """

    def __init__(self, classifier, batch_size, assume_fitted=False, verbose=0, **kmeans_args):
        super().__init__()
        # TODO: can we check that the classifier has a predict_proba?
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted

        self.kmeans = KMeansSampler(
            batch_size,
            verbose, **kmeans_args)
        self.batch_size = batch_size
        self.verbose = verbose
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True


    def fit(self, X, y):
        """Fit the estimator on labeled samples.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Training data
            y (numpy array, shape (n_samples,)): Target values

        Returns:
            self: An instance of self.
        """
        if not self.assume_fitted:
            self.classifier_.fit(X, y)
        return self

    def predict(self, X):
        proba = _get_probability_classes(self.classifier_, X)
        uncertainty = 1 - np.max(proba, axis=1)
        selected = self.kmeans.predict(proba, sample_weight=uncertainty)
        return selected.astype(int)