from sklearn.model_selection import ShuffleSplit

from .typeutils import RandomStateType


class RepeatedHalfSplit(ShuffleSplit):
    """Repeated Half Split cross-validator

    Provides train/test indices to split data in train/test sets. This strategy
    splits the data in two halves and returns each split as train and test
    (and vice versa for the next iteration). It can therefore only work with
    even number of folds.
    Accordind to Diterrich 2017, this method is the most powerful repetition
    scheme with acceptable Type I error.  
    
    Args:
        n_splits: Number of folds, must be even.

    Example:
        >>> import numpy as np
        >>> from sklearn.model_selection import LeavePOut
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([1, 2, 3, 4])
        >>> rhscv = RandomHalfSplit(4)
        >>> for train_index, test_index in rhscv.split(X):
        ...     print("TRAIN:", train_index, "TEST:", test_index)
        ...     X_train, X_test = X[train_index], X[test_index]
        ...     y_train, y_test = y[train_index], y[test_index]
        TRAIN: [2 3] TEST: [0 1]
        TRAIN: [0 1] TEST: [2 3]
        TRAIN: [3 0] TEST: [1 2]
        TRAIN: [1 2] TEST: [3 0]

    Notes:
        [1] "Approximate statistical tests for comparing supervised classification
        learning algorithms." by T. Dietterich.
        (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.3325&rep=rep1&type=pdf)
    """

    def __init__(self, n_splits: int, random_state: RandomStateType = None):
        if (n_splits % 2) == 1:
            raise ValueError('Number of folds must be even')
        super().__init__(
            n_splits=n_splits // 2,
            test_size=.5,
            train_size=None,
            random_state=random_state)

    def _iter_indices(self, X, y=None, groups=None):
        for train, test in super()._iter_indices(X, y=y, groups=groups):
            yield train, test
            yield test, train