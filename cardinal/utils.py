from typing import Union

import numpy as np
from sklearn.utils.validation import _num_samples
from sklearn.utils import _safe_indexing, indexable
from sklearn.model_selection import train_test_split

from .typeutils import check_random_state, RandomStateType


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
    """Indexer for train, test, selected, and batches.

    Active learning implies to selected subset of samples from other subsets
    which makes indexing difficult. This class allows easy indxing.

    Args:
        arrays: Allowed inputs are lists, numpy arrays, scipy-sparse matrices
            or pandas dataframes.
        test_size: If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If not specified, it is set to 0 (no test
            set).
        train_size: If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.
        random_state: Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
        shuffle: Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None.
        stratify: If not None, data is split in a stratified fashion, using this as
            the class labels.
    """
    def __init__(
        self, 
        *arrays,
        test_size: Union[float, int]=0,
        train_size: Union[float, int]=None,
        random_state: RandomStateType=None,
        shuffle: boolean=True,
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

    TRAIN_UNSELECTED = -1
    TEST = -2

    def add_batch(self, indices):
        """Add indices of a new batch to selected samples

        Args:
            indices: Arrays of indices of selected samples
        """
        if self.current_iter is None:
            self.current_iter = -1
        self.current_iter += 1
        self._mask[np.where(self._mask == self.TRAIN_UNSELECTED)[0][indices]] = self.current_iter

    def get_batch(self, iter: int=None):
        """Get indices of the last batch, or from a previous one.

        Args:
            iter: Iteration of the desired batch. Default (None) returns
                the last one.

        Returns:
            Indices of the batch corresponding to the iteration.
        """
        if iter is None:
            iter = self.current_iter

        index = (self._mask == iter)
        return [_safe_indexing(a, index) for a in self.arrays]

    def get_selected(self, iter: int=None):
        """Get indices of the samples selected so far.

        Args:
            iter: Iteration of the desired samples. Default (None) returns
                the last one.

        Returns:
            Indices selected until the given iteration.
        """
        index = (self._mask >= 0)  # A bit unsafe but simpler
        if iter is not None:
            index = np.logical_and(index, self._mask <= iter)
        return [_safe_indexing(a, index) for a in self.arrays]

    def get_non_selected(self, iter: int=None):
        """Get indices of the samples not selected so far.

        Args:
            iter: Iteration of the desired samples. Default (None) returns
                the last one.

        Returns:
            Indices not selected until the given iteration.
        """
        index = (self._mask == self.TRAIN_UNSELECTED)
        if iter is not None:
            index = np.logical_or(index, self._mask > iter)
        return [_safe_indexing(a, index) for a in self.arrays]

    def get_train(self):
        """Get indices of the train samples.

        Returns:
            List of indices selected for train.
        """
        index = (self._mask != self.TEST)
        return [_safe_indexing(a, index) for a in self.arrays]

    def get_test(self):
        """Get indices of the test samples.

        Returns:
            List of indices selected for test.
        """
        index = (self._mask == self.TEST)
        return [_safe_indexing(a, index) for a in self.arrays]