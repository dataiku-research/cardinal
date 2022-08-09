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
        n_samples: Number of samples in total
        test_index: Index of test samples if any
    """

    # Constructors
    ##############

    def __init__(self, n_samples, test_index=None, dtype=np.int8):
        self._mask = np.full(n_samples, self.TRAIN_UNSELECTED, dtype=dtype)
        self.current_iter = None
        if test_index is not None:
            self._mask[test_index] = self.TEST

    @classmethod
    def train_test_split(
        cls,
        n_samples: int,
        test_size: Union[float, int]=0,
        train_size: Union[float, int]=None,
        random_state: RandomStateType=None,
        shuffle: bool=True,
        stratify=None,
        dtype=np.int8
    ):
        """Create an indexer from train_test_split

        Args:
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
        test_index = None
        if test_size != 0:
            random_state = check_random_state(random_state)
            _, test_index = train_test_split(
                np.arange(n_samples),
                test_size=test_size,
                train_size=train_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify)
        return cls(n_samples, test_index=test_index, dtype=dtype)

    @classmethod
    def from_mask(
        cls,
        mask
    ):
        """Create an indexer from train_test_split

        Args:
            mask: numpy int array
                A valid mask for active learning splitter.
        """
        max_iter = np.max(mask)

        assert(np.all(mask >= -2))
        assert(np.unique(mask).shape[0] == max_iter + 3)

        splitter = cls(mask.shape[0])
        splitter._mask = mask

        splitter.current_iter = max_iter

        return splitter

    # Constants
    ###########

    TRAIN_UNSELECTED = -1
    TEST = -2

    # Initilization
    ###############

    def initialize_with_random(self, n_init_samples:int, at_least_one_of_each_class=None,
                               random_state: RandomStateType=None,):
        """Initialize a splitter with random samples.
        
        Args:
            n_init_samples: Number of samples in the first selection
            at_least_one_of_each_class: If specified, first iteration contains at least
                                        one sample from each class.
        """
        random_state = check_random_state(random_state)

        if self.current_iter is not None:
            raise ValueError('Initilization must be performed after creation of the splitter')

        if at_least_one_of_each_class is not None:
            if at_least_one_of_each_class.shape[0] != self.train.sum():
                raise ValueError('Labels provided must have the same shape as train')
            # We chose at least one index per class
            inverse_table = np.unique(at_least_one_of_each_class, return_inverse=True)[1]
            shuffled_index = np.arange(inverse_table.shape[0])
            random_state.shuffle(shuffled_index)
            one_per_class = np.unique(inverse_table[shuffled_index], return_index=True)[1]
            one_per_class = shuffled_index[one_per_class]
            self._mask[one_per_class] = 0
            n_init_samples -= one_per_class.shape[0]

        indices = random_state.choice(np.where(self.train)[0], replace=False, size=n_init_samples)
        self._mask[indices] = 0
        self.current_iter = 0


    def initialize_with_indices(self, indices):
        """Initialize a splitter with a given list of indices.
        
        Args:
            indices: List of indices for the initialization
        """
        indices = self.dereference_batch_indices(indices)
        self._mask[indices] = 0
        self.current_iter = 0


    # Accessors
    ###########

    def _check_init(self):
        if self.current_iter is None:
            raise ValueError('Splitter must be initialized before any index is accessed')


    @property
    def selected(self):
        return self.selected_at(None)

    def selected_at(self, iter: int=None):
        """Get indices of the samples selected so far.

        Args:
            iter: Iteration of the desired samples. Default (None) returns
                the last one.

        Returns:
            Indices selected until the given iteration.
        """
        self._check_init()
        index = (self._mask >= 0)  # A bit unsafe but simpler
        if iter is not None:
            index = np.logical_and(index, self._mask <= iter)
        return index

    def dereference_batch_indices(self, indices):
        return np.where(self._mask == self.TRAIN_UNSELECTED)[0][indices]

    def add_batch(self, indices):
        """Add indices of a new batch to selected samples

        Args:
            indices: Arrays of indices of selected samples
        """
        self._check_init()
        next_iter = self.current_iter + 1
        self._mask[self.dereference_batch_indices(indices)] = next_iter
        self.current_iter = next_iter

    @property
    def batch(self):
        return self.batch_at(None)

    def batch_at(self, iter: int):
        """Get indices of the last batch, or from a previous one.

        Args:
            iter: Iteration of the desired batch. Default (None) returns
                the last one.

        Returns:
            A binary mask of the batch samples at a given iteration.
        """
        self._check_init()
        if iter is None:
            iter = self.current_iter - 1

        batch_mask = (self._mask == (iter + 1))
        if not batch_mask.any():
            raise ValueError('Asking for a batch that has not been computed yet')
        return batch_mask

    @property
    def non_selected(self):
        return self.non_selected_at(None)

    def non_selected_at(self, iter: int):
        """Get indices of the samples not selected so far.

        Args:
            iter: Iteration of the desired samples. Default (None) returns
                the last one.

        Returns:
            Indices not selected until the given iteration.
        """
        self._check_init()
        index = (self._mask == self.TRAIN_UNSELECTED)
        if iter is not None:
            index = np.logical_or(index, self._mask > iter)
        return index

    @property
    def train(self):
        """Get indices of the train samples.

        Returns:
            A binary mask of the train samples.
        """
        return (self._mask != self.TEST)

    @property
    def test(self):
        """Get indices of the test samples.

        Returns:
            A binary mask of the test samples.
        """
        return (self._mask == self.TEST)
