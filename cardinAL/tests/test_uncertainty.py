import numpy as np
from numpy.testing import assert_array_equal

from cardinAL.uncertainty import ConfidenceSampler, MarginSampler, EntropySampler


def test_all_uncertainty():

    proba = np.array([[0.10, 0.20, 0.30, 0.40],
                      [0.00, 0.05, 0.55, 0.60],
                      [0.15, 0.45, 0.20, 0.20 ]])

    # Confidence sampling choses the first sample
    sampler = ConfidenceSampler('precomputed', 1, assume_fitted=True)
    sampler.fit([], [])  # No training set, we consider it precomputed
    selected = sampler.predict(proba)
    assert_array_equal(selected, np.array([1, 0, 0]))

    # Margin sampling choses the second sample
    sampler = MarginSampler('precomputed', 1, assume_fitted=True)
    sampler.fit([], [])  # No training set, we consider it precomputed
    selected = sampler.predict(proba)
    assert_array_equal(selected, np.array([0, 1, 0]))

    # Entropy sampling choses the third sample
    sampler = EntropySampler('precomputed', 1, assume_fitted=True)
    sampler.fit([], [])  # No training set, we consider it precomputed
    selected = sampler.predict(proba)
    assert_array_equal(selected, np.array([0, 0, 1]))