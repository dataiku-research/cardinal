import numpy as np

from .typeutils import check_random_state
from sklearn.utils.validation import _num_samples
from sklearn.utils import _safe_indexing, indexable
from sklearn.model_selection import train_test_split


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

    def __init__(self, size: int):
        self.size = size
        self._mask = np.zeros((size,), dtype=np.bool)
        self._indices = np.arange(size)

    def add_to_selected(self, indices):
        self._mask[self._indices[~self._mask][indices]] = True

    @property
    def selected(self):
        v = self._mask.copy().view()
        v.setflags(write=False)
        return v

    @property
    def non_selected(self):
        v = (~self._mask).view()
        v.setflags(write=False)
        return v


class ActiveLearningSplitter():

    TRAIN_UNSELECTED = -1
    TEST = -2

    def __init__(
        self, 
        *arrays,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
        dtype=np.int8
    ):
        n_samples = _num_samples(arrays[0])
        self._mask = np.full(n_samples, self.TRAIN_UNSELECTED, dtype=dtype)
        if test_size != 0:
            self.random_state = check_random_state(random_state)
            _, test = train_test_split(
                np.arange(n_samples),
                test_size=test_size,
                train_size=train_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify)
            self._mask[test] = self.TEST
        self.arrays = arrays
        self.current_iter = None

    def add_batch(self, indices):
        if self.current_iter is None:
            self.current_iter = -1
        self.current_iter += 1
        self._mask[np.where(self._mask == self.TRAIN_UNSELECTED)[0][indices]] = self.current_iter

    def get_batch(self, iter=None):
        if iter is None:
            iter = self.current_iter

        index = (self._mask == iter)
        return [_safe_indexing(a, index) for a in self.arrays]

    def get_selected(self):
        index = (self._mask >= 0)  # A bit unsafe but simpler
        return [_safe_indexing(a, index) for a in self.arrays]

    def get_non_selected(self):
        index = (self._mask == self.TRAIN_UNSELECTED)
        return [_safe_indexing(a, index) for a in self.arrays]

    def get_train(self):
        index = (self._mask != self.TEST)
        return [_safe_indexing(a, index) for a in self.arrays]

    def get_test(self):
        index = (self._mask == self.TEST)
        return [_safe_indexing(a, index) for a in self.arrays]