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

    part = np.partition(-classwise_uncertainty, 1, axis=1)
    margin = part[:, 0] - part[:, 1]
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

