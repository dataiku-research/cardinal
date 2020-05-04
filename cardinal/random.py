import numpy as np

from .base import ScoredQuerySampler
from .typeutils import RandomStateType, check_random_state


class RandomSampler(ScoredQuerySampler):
    """Randomly select samples

    Args:
        batch_size : Number of samples to select.
        random_state : The seed of the pseudo random number generator to use
            when shuffling the data.  If int, random_state is the seed used by
            the random number generator; If RandomState instance, random_state
            is the random number generator; If None (defdault), the random
            number generator is the RandomState instance used by `np.random`.

    Attributes:
        random_state : The random state used by the sampler.
    """

    def __init__(self, batch_size: int, random_state: RandomStateType = None):
        super().__init__(batch_size=batch_size)
        self.random_state = random_state

    def fit(self, X: np.array = None, y: np.array = None) -> 'RandomSampler':
        """Sets the random state
        
        Args:
            X: Labeled samples of shape (n_samples, n_features).
            y: Labels of shape (n_samples).
        
        Returns:
            The object itself
        """
        self.random_state = check_random_state(self.random_state)
        return self

    def score_samples(self, X: np.array) -> np.array:
        return self.random_state.rand(X.shape[0])
