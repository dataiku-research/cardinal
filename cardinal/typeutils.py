from typing import Union, Callable
import numpy as np


RandomStateType = Union[np.random.RandomState, int, None]
MetricType = Union[Callable, str]


def check_random_state(seed: RandomStateType):
    """Turn seed into a np.random.RandomState instance
    
    Args:
    seed : If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.

    Note
    ----
    This was taken from scikit-learn and slightly modified
    """
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.mtrand._rand


class NotEnoughSamplesWarning(UserWarning):
    """Custom warning used when a sampler is given less than batch_size samples
    """


def _has_method(obj, method):
    return hasattr(obj, method) and callable(getattr(obj, method))


def check_proba_estimator(obj):

    predict_name = 'predict_proba'

    # Special case, we allow keras models
    package = obj.__class__.__module__.split('.')[0]

    if package == 'keras':
        predict_name = 'predict'

    # If the object has the right method, it's ok
    has_fit = _has_method(obj, 'fit')
    has_predict_proba = _has_method(obj, predict_name)

    if has_fit and has_predict_proba:
        return

    origin = "object from package " + package

    if package == "__main__":
        origin = "user-defined object"

    raise TypeError('Provided {} does not have required methods fit and {}.'
                    ''.format(origin, predict_name))