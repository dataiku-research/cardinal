import numpy as np

from cardinal.clustering import KCenterGreedy


def test_k_center_greedy():

    # Those points are crafted so that they are selected in order by
    # k center greedy, no matter batch size

    X = np.array([
        [ 0, 0],
        [64, 1],
        [32, 0],
        [16, 1],
        [ 8, 0],
        [ 4, 1],
    ])

    # Let us consider the first sample selected, and select the other by batch of 2
    selected = np.zeros(X.shape[0], dtype=bool)
    selected[0] = True
    sampler = KCenterGreedy(lambda x:x, 2)

    for i in range((X.shape[0] - 1) // 2):
        sampler.fit(X[selected], None)
        indices = sampler.select_samples(X[~selected])
        assert(indices[0] == 0)
        assert(indices[1] == 1)
        selected[indices] = True
