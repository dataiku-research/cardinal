from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from scipy.stats import entropy
import numpy as np
from keras.models import Model


def _get_probability_classes(classifier, X):
    try:
        if isinstance(classifier, Model):  # Keras models have no predict_proba
            classwise_uncertainty = classifier.predict(X)
        else:  # sklearn model
            classwise_uncertainty = classifier.predict_proba(X)
    except NotFittedError:
        raise
    return classwise_uncertainty


def uncertainty_sampling(classifier: BaseEstimator, X: np.ndarray,
                         n_instances: int = 1) -> np.ndarray:
    """
    Uncertainty sampling query strategy. Selects the least sure instances for labelling.
    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    classwise_uncertainty = _get_probability_classes(classifier, X)
        
    # for each point, select the maximum uncertainty
    uncertainty = 1 - np.max(classwise_uncertainty, axis=1)
    index = np.flip(np.argsort(uncertainty))[:n_instances]
    
    return index, uncertainty[index]

def margin_sampling(classifier: BaseEstimator, X: np.ndarray,
                    n_instances: int = 1) -> np.ndarray:
    """
    Margin sampling query strategy. Selects the instances where the difference between
    the first most likely and second most likely classes are the smallest.
    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    classwise_uncertainty = _get_probability_classes(classifier, X)

    part = np.partition(classwise_uncertainty, -2, axis=1)
    margin = 1 - (part[:, -1] - part[:, -2])
    index = np.flip(np.argsort(margin))[:n_instances]
    
    return index, margin[index]

def entropy_sampling(classifier: BaseEstimator, X: np.ndarray,
                     n_instances: int = 1) -> np.ndarray:
    """
    Entropy sampling query strategy. Selects the instances where the class probabilities
    have the largest entropy.
    
    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    classwise_uncertainty = _get_probability_classes(classifier, X)
    
    entropies = np.transpose(entropy(np.transpose(classwise_uncertainty)))
    index = np.flip(np.argsort(entropies))[:n_instances]
    
    return index, entropies[index]


class UncertaintySampler(BaseQuerySampler):
    """Selects samples with lowest prediction confidence.

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

    def __init__(self, pipeline, batch_size, verbose=0):
        super().__init__()
        # TODO: can we check that the pipeline has a predict_proba?
        self.pipeline_ = pipeline
        self.batch_size = batch_size
        self.verbose = verbose

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
        
        # We delegate pretty much everything to the estimator
        self.pipeline_.fit(X, y)
        
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
        index, confidence = uncertainty_sampling(self.pipeline_, X, n_instances=X.shape[0])
        
        self.confidence_ = confidence
        index = index[:self.batch_size]
        
        selected_samples[index] = 1
        self.labels_ = selected_samples

        return selected_samples


class MarginSampler(BaseQuerySampler):
    """Selects samples with the lowest margin between the first and second classes.

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

    def __init__(self, pipeline, batch_size, verbose=0):
        super().__init__()
        # TODO: can we check that the pipeline has a predict_proba?
        self.pipeline_ = pipeline
        self.batch_size = batch_size
        self.verbose = verbose

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
        
        # We delegate pretty much everything to the estimator
        self.pipeline_.fit(X, y)
        
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
        index, confidence = margin_sampling(self.pipeline_, X, n_instances=X.shape[0])
        
        self.confidence_ = confidence
        index = index[:self.batch_size]
        
        selected_samples[index] = 1
        self.labels_ = selected_samples

        return selected_samples


class EntropySampler(BaseQuerySampler):
    """Selects samples with highest entropy.

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

    def __init__(self, pipeline, batch_size, verbose=0):
        super().__init__()
        # TODO: can we check that the pipeline has a predict_proba?
        self.pipeline_ = pipeline
        self.batch_size = batch_size
        self.verbose = verbose

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
        
        # We delegate pretty much everything to the estimator
        self.pipeline_.fit(X, y)
        
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
        index, confidence = entropy_sampling(self.pipeline_, X, n_instances=X.shape[0])
        
        self.confidence_ = confidence
        index = index[:self.batch_size]
        
        selected_samples[index] = 1
        self.labels_ = selected_samples

        return selected_samples
