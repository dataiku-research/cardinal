import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cardinal.uncertainty import ConfidenceSampler, MarginSampler, EntropySampler


def test_all_uncertainty():

    proba = np.array([[0.10, 0.20, 0.30, 0.40],
                      [0.00, 0.05, 0.45, 0.50],
                      [0.15, 0.45, 0.20, 0.20]])

    # Confidence sampling choses the first sample
    sampler = ConfidenceSampler('precomputed', 1, assume_fitted=True)
    sampler.fit([], [])  # No training set, we consider it precomputed
    selected = sampler.select_samples(proba)
    assert_array_equal(selected, np.array([0]))

    # Margin sampling choses the second sample
    sampler = MarginSampler('precomputed', 1, assume_fitted=True)
    sampler.fit([], [])  # No training set, we consider it precomputed
    selected = sampler.select_samples(proba)
    assert_array_equal(selected, np.array([1]))

    # Entropy sampling choses the third sample
    sampler = EntropySampler('precomputed', 1, assume_fitted=True)
    sampler.fit([], [])  # No training set, we consider it precomputed
    selected = sampler.select_samples(proba)
    assert_array_equal(selected, np.array([2]))


class WithAllMethods:

    def fit(X, y=None):
        pass

    def predict_proba(X):
        pass


class MissingMethods:

    def fit(X, y=None):
        pass


def test_types():
    ConfidenceSampler(WithAllMethods(), 1)
    with pytest.raises(TypeError):
        ConfidenceSampler(MissingMethods(), 1)
