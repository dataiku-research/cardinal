import numpy as np

from .typeutils import check_random_state


def pad_with_random(array, size, min, max, random_state=None):
    n_missing = size - len(array)
    if n_missing <= 0:
        return array
    random_state = check_random_state(random_state)
    choices = np.arange(min, max)
    mask = np.ones(max - min, dtype=bool)
    mask[array - min] = False
    choices = choices[mask]
    padding = random_state.choice(choices, n_missing)
    return np.concatenate([array, padding])
