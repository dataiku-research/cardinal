import numpy as np

from cardinal.clustering import KCenterGreedy
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
import abc


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


def test_k_center_greedy_duplicates():

    # In case of duplicates KCenter relies on random selection.

    X = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
    ])

    sampler = KCenterGreedy(lambda x:x, 4)

    sampler.fit(X[[0]], None)
    indices = sampler.select_samples(X[1:])
    assert(indices.shape[0] == 4)


###################################################################################
# THIS IS THE REFERENCE IMPLEMENTATION OF KCENTER GREEDY
# Our approach should be slightlyh more optimized but have the exact same outcome.
# I put it here for testing purposes.

class kCenterGreedy():

    def __init__(self, X,  metric='euclidean'):
        self.X = X
        # self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.max_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []


    def flatten_X(self):
      shape = self.X.shape
      flat_X = self.X
      if len(shape) > 2:
        flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
      return flat_X

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
          self.min_distances = None
        if only_new:
          cluster_centers = [d for d in cluster_centers
                            if d not in self.already_selected]
        if cluster_centers:
          x = self.features[cluster_centers]
          # Update min_distances for all examples given new cluster center.
          dist = pairwise_distances(self.features, x, metric=self.metric)#,n_jobs=4)

          if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1,1)
          else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch(self, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
          # Assumes that the transform function takes in original data and not
          # flattened data.
          print('Getting transformed features...')
        #   self.features = model.transform(self.X)
          print('Calculating distances...')
          self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
          print('Using flat_X as features.')
          self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
          if self.already_selected is None:
            # Initialize centers with a randomly selected datapoint
            ind = np.random.choice(np.arange(self.n_obs))
          else:
            ind = np.argmax(self.min_distances)
          # New examples should not be in already selected since those points
          # should have min_distance of zero to a cluster center.
          assert ind not in already_selected

          self.update_distances([ind], only_new=True, reset_dist=False)
          new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
                % max(self.min_distances))


        self.already_selected = already_selected

        return new_batch


def test_k_center_greedy_against_ref():

    from cardinal.clustering import KCenterGreedy

    X = np.random.random((100, 10))
    sampler = kCenterGreedy(X)
    ref_selection = np.asarray(sampler.select_batch(np.arange(10), 10))

    sampler = KCenterGreedy(lambda x:x, 10)
    sampler.fit(X[np.arange(10)])
    our_selection = sampler.select_samples(X[10:]) + 10

    assert((ref_selection == our_selection).all())