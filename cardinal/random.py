import numpy as np


def random_sampling(X: np.ndarray,
                    idx_labeled: np.ndarray = None,
                    n_instances: int = 1) -> np.ndarray:
    """
    Random sampling query strategy. Selects instances randomly
    
    Args:
        X: The pool of samples to query from.
        idx_labeled: Samples to remove because they have been already labelled
        n_instances: Number of samples to be queried.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
        
    Note:
        This class is handy for testing against naive method
    """
    unlabeled = np.ones(n_instances)
    if idx_labeled is not None:
        unlabeled[idx_labeled] = 0
    index = np.where(unlabeled)[0]
    np.random.shuffle(index)
    index = index[:n_instances]
    
    return index, unlabeled[index]