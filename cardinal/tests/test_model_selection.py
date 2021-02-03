from cardinal.model_selection import RepeatedHalfSplit
import pytest
import numpy as np
from numpy.testing import assert_array_equal


def test_repeated_half_split():
    with pytest.raises(ValueError):
        RepeatedHalfSplit(3)

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    all_splits = list(RepeatedHalfSplit(4).split(X))
    for i in range(2):
        tr1, te1 = all_splits[i * 2]
        tr2, te2 = all_splits[i * 2 + 1]
        assert_array_equal(te2, tr1)
        assert_array_equal(te1, tr2)