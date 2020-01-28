import copy

from sklearn.datasets import make_classification

from cardinAL import 

X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=2,
                           n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0,
                           hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)