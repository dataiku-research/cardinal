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


class SampleSelector():

    def __init__(self, size, init=None, cache=None, cache_name='selected'):
        self.size = size
        self._mask = np.zeros((size,), dtype=np.bool)
        self._indices = np.arange(size)
        if init:
            self.add_to_selected(init)
        self._persisted_value = None
        if cache:
            self._persisted_value = cache._persisted_value(name, self._mask)

    def add_to_selected(self, indices):
        self._mask[self._indices[~self._mask][indices]] = True
        if self._persisted_value:
            self._persisted_value.set(self._mask)

    @property
    def selected(self):
        v = self._mask.view()
        v.setflags(write=False)
        return v

    @property
    def non_selected(self):
        v = (~self._mask).view()
        v.setflags(write=False)
        return v

    def resume(self, mask):
        self._mask = mask.copy()