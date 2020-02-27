from sklearn.exceptions import NotFittedError
from scipy.stats import entropy
from sklearn.base import BaseEstimator
import numpy as np
from .base import BaseQuerySampler


def _get_probability_classes(classifier, X):
    if classifier == 'precomputed':
        return X
    elif classifier.__class__.__module__.split('.')[0] == 'keras':  # Keras models have no predict_proba
        classwise_uncertainty = classifier.predict(X)
    else:  # sklearn compatible model
        classwise_uncertainty = classifier.predict_proba(X)
    return classwise_uncertainty


def confidence_score(classifier: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Measure the confidence score of a model for a set of samples.

    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.

    Returns:
        The confidence score for each sample.
    """
    classwise_uncertainty = _get_probability_classes(classifier, X)
    uncertainty = 1 - np.max(classwise_uncertainty, axis=1)
    return uncertainty


def margin_score(classifier: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Compute the difference between the two top probability classes for each sample. 

    This strategy takes the probabilities of top two classes and uses their
    difference as a score for selection.

    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.

    Returns:
        The margin score for each sample.
    """
    classwise_uncertainty = _get_probability_classes(classifier, X)
    part = np.partition(classwise_uncertainty, -2, axis=1)
    margin = 1 - (part[:, -1] - part[:, -2])
    return margin


def entropy_score(classifier: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Entropy sampling query strategy, uses entropy of all probabilities as score.

    This strategy selects the samples with the highest entropy in their prediction
    probabilities.
    
    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.

    Returns:
        The entropy score for each label
    """
    classwise_uncertainty = _get_probability_classes(classifier, X)
    entropies = np.transpose(entropy(np.transpose(classwise_uncertainty)))
    return entropies


class ConfidenceSampler(BaseQuerySampler):
    """Selects samples with lowest prediction confidence.

    Lowest confidence sampling looks at the probability of the class predicted by
    the classifier and selects the samples where this probability is the lowest.

    Parameters:
        classifier (sklearn.BaseEstimator): Classifier used to
            determine the prediction confidence. The object must
            comply with scikit-learn interface and expose a
            `predict_proba` method.
        batch_size (int): Number of samples to draw when predicting.
        assume_fitted (bool): If true, classifier is not refit
        verbose (int, optional): The verbosity level. Defaults to 0.
    
    Attributes:
        classifier_ (sklearn.BaseEstimator): The fitted classifier.
    """

    def __init__(self, classifier, batch_size, assume_fitted=False, verbose=0):
        super().__init__(batch_size=batch_size)
        # TODO: can we check that the classifier has a predict_proba?
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
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

    def score_samples(self, X):
        """Selects the samples to annotate from unlabelled data.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Samples to evaluate.

        Returns:
            predictions (np.array): The score of each sample according to lowest confidence estimation.
        """
        return confidence_score(self.classifier_, X)


class MarginSampler(BaseQuerySampler):
    """Selects samples with greatest confusion between the top two classes.

    Smallest margin sampling uses the difference of predicted probability between
    the top two classes to select the samples on which the model is hesitating
    the most, hence the lowest difference.

    Parameters:
        classifier (sklearn.BaseEstimator): Classifier used to
            determine the prediction confidence. The object must
            comply with scikit-learn interface and expose a
            `predict_proba` method.
        batch_size (int): Number of samples to draw when predicting.
        assume_fitted (bool): If true, classifier is not refit
        verbose (int, optional): The verbosity level. Defaults to 0.
    
    Attributes:
        classifier_ (sklearn.BaseEstimator): The fitted classifier.
    """

    def __init__(self, classifier, batch_size, assume_fitted=False, verbose=0):
        super().__init__(batch_size=batch_size)
        # TODO: can we check that the classifier has a predict_proba?
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
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

    def score_samples(self, X):
        """Selects the samples to annotate from unlabelled data.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Samples to evaluate.

        Returns:
            predictions (np.array): The score of each sample according to smallest margin estimation.
        """
        return margin_score(self.classifier_, X)


class EntropySampler(BaseQuerySampler):
    """Selects samples with greatest entropy among all class probabilities.

    Greatest entropy sampling measures the uncertainty of the model over all
    classes through the entropy of the probabilites of all classes. Highest
    entropy samples are selected.

    Parameters:
        classifier (sklearn.BaseEstimator): Classifier used to
            determine the prediction confidence. The object must
            comply with scikit-learn interface and expose a
            `predict_proba` method.
        batch_size (int): Number of samples to draw when predicting.
        assume_fitted (bool): If true, classifier is not refit
        verbose (int, optional): The verbosity level. Defaults to 0.
    
    Attributes:
        classifier_ (sklearn.BaseEstimator): The fitted classifier.
    """

    def __init__(self, classifier, batch_size, assume_fitted=False, verbose=0):
        super().__init__(batch_size=batch_size)
        # TODO: can we check that the classifier has a predict_proba?
        self.classifier_ = classifier
        self.assume_fitted = assume_fitted
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

    def score_samples(self, X):
        """Selects the samples to annotate from unlabelled data.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Samples to evaluate.

        Returns:
            predictions (np.array): The entropy score of each sample.
        """
        return entropy_score(self.classifier_, X)
