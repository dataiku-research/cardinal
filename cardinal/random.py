import numpy as np
from .base import BaseQuerySampler


def random_sampling(X: np.ndarray,
                    idx_labeled: np.ndarray = None,
                    n_instances: int = 1) -> np.ndarray:
    """
    Random sampling query strategy. Selects instances randomly
    
    Args:
        X: The pool of samples to query from.
        idx_labeled: Samples to remove because they have been already labelled
        n_instances: Number of samples to be queried.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
        
    Note:
        This class is handy for testing against naive method
    """
    unlabeled = np.ones(n_instances)
    if idx_labeled is not None:
        unlabeled[idx_labeled] = 0
    index = np.where(unlabeled)[0]
    np.random.shuffle(index)
    index = index[:n_instances]
    
    return index, unlabeled[index]


class RandomSampler(BaseQuerySampler):
    """TODO
    Documentation.

    Parameters
    ----------
    batch_size : float, default: 0.05
        If batch_size < 1., it is interpreted as fraction of the training
        set to select for labeling. If batch_size >= 1., it is interpreted
        as the number of samples to draw.
    shuffle : bool, optional
        Whether or not the training data should be shuffled after each epoch.
        Defaults to True.
    verbose : integer, optional
        The verbosity level
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.
    class_weight : dict, {class_label: weight} or "balanced" or None, optional
        Preset for the class_weight fit parameter.
        Weights associated with classes. If not given, all classes
        are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
    Attributes
    ----------
    pipeline_ : sklearn.pipeline
        Pipeline used to predict the class probability.
    """

    def __init__(self, batch_size=0.05, shuffle=True,
                 verbose=0, random_state=None,
                 n_iter_no_change=5, class_weight=None):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight

    def fit(self, X, y):
        """Fit linear model with Stochastic Gradient Descent.
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
        self._classes = ['not selected', 'selected']
        self.random_state = check_random_state(self.random_state)
        return self

    def predict(self, X):
        selected_samples = np.zeros(X.shape[0])
        selected_samples[:self.batch_size] = 1
        self.random_state.shuffle(selected_samples)
        return selected_samples