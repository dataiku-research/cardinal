from typing import Union
import numpy as np


RandomStateType = Union[np.random.RandomState, int, None]


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
