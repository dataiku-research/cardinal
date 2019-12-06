from sklearn.exceptions import NotFittedError
from scipy.stats import entropy
from sklearn.base import BaseEstimator
import numpy as np
from keras.models import Model
from .base import BaseQuerySampler


def _get_probability_classes(classifier, X):
    try:
        if isinstance(classifier, Model):  # Keras models have no predict_proba
            classwise_uncertainty = classifier.predict(X)
        else:  # sklearn model
            classwise_uncertainty = classifier.predict_proba(X)
    except NotFittedError:
        raise
    return classwise_uncertainty


def confidence_sampling(classifier: BaseEstimator, X: np.ndarray,
                        n_instances: int = 1) -> np.ndarray:
    """Lowest confidence sampling query strategy. Selects the least sure instances for labelling.

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
    """Margin sampling query strategy, selects the samples with lowest difference between top 2 probabilities.

    This strategy takes the probabilities of top two classes and uses their
    difference as a score for selection.

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
    """Entropy sampling query strategy, uses entropy of all probabilities as score.

    This strategy selects the samples with the highest entropy in their prediction
    probabilities.
    
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
        verbose (int, optional): The verbosity level. Defaults to 0.
    
    Attributes:
        classifier_ (sklearn.BaseEstimator): The fitted classifier.
    """

    def __init__(self, classifier, batch_size, verbose=0):
        super().__init__()
        # TODO: can we check that the classifier has a predict_proba?
        self.classifier_ = classifier
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the estimator on labeled samples.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Training data
            y (numpy array, shape (n_samples,)): Target values

        Returns:
            self: An instance of self.
        """
        # We delegate pretty much everything to the estimator
        self.classifier_.fit(X, y)
        
        return self

    def predict(self, X):
        """Selects the samples to annotate from unlabelled data.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Samples to evaluate.

        Returns:
            predictions (np.array): Returns an array where selected samples are classified as 1.
        """
        selected_samples = np.zeros(X.shape[0])
        index, confidence = confidence_sampling(self.classifier_, X, n_instances=X.shape[0])
        
        self.confidence_ = confidence
        index = index[:self.batch_size]
        
        selected_samples[index] = 1
        self.labels_ = selected_samples

        return selected_samples


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
        verbose (int, optional): The verbosity level. Defaults to 0.
    
    Attributes:
        classifier_ (sklearn.BaseEstimator): The fitted classifier.
    """

    def __init__(self, classifier, batch_size, verbose=0):
        super().__init__()
        # TODO: can we check that the classifier has a predict_proba?
        self.classifier_ = classifier
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the estimator on labeled samples.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Training data
            y (numpy array, shape (n_samples,)): Target values

        Returns:
            self: An instance of self.
        """
        # We delegate pretty much everything to the estimator
        self.classifier_.fit(X, y)
        
        return self

    def predict(self, X):
        """Selects the samples to annotate from unlabelled data.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Samples to evaluate.

        Returns:
            predictions (np.array): Returns an array where selected samples are classified as 1.
        """
        selected_samples = np.zeros(X.shape[0])
        index, confidence = margin_sampling(self.classifier_, X, n_instances=X.shape[0])
        
        self.confidence_ = confidence
        index = index[:self.batch_size]
        
        selected_samples[index] = 1
        self.labels_ = selected_samples

        return selected_samples


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
        verbose (int, optional): The verbosity level. Defaults to 0.
    
    Attributes:
        classifier_ (sklearn.BaseEstimator): The fitted classifier.
    """

    def __init__(self, classifier, batch_size, verbose=0):
        super().__init__()
        # TODO: can we check that the classifier has a predict_proba?
        self.classifier_ = classifier
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the estimator on labeled samples.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Training data
            y (numpy array, shape (n_samples,)): Target values

        Returns:
            self: An instance of self.
        """
        # We delegate pretty much everything to the estimator
        self.classifier_.fit(X, y)
        
        return self

    def predict(self, X):
        """Selects the samples to annotate from unlabelled data.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Samples to evaluate.

        Returns:
            predictions (np.array): Returns an array where selected samples are classified as 1.
        """
        selected_samples = np.zeros(X.shape[0])
        index, confidence = entropy_sampling(self.classifier_, X, n_instances=X.shape[0])
        
        self.confidence_ = confidence
        index = index[:self.batch_size]
        
        selected_samples[index] = 1
        self.labels_ = selected_samples

        return selected_samples
