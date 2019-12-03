import numpy as np
from .base import BaseQuerySampler
from sklearn.base import clone
from sklearn.exceptions import NotFittedError


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