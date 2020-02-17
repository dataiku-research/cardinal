import copy
import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

from cardinAL.alipy import AlipyWrapper
#from cardinAL.modal import ModalWrapper
from modAL.density import information_density
from cardinAL.libact import LibactWrapper
from cardinAL.random import RandomSampler
from cardinAL.plotting import plot_confidence_interval

from alipy.query_strategy.query_labels import QueryInstanceGraphDensity
from libact.query_strategies import DensityWeightedMeta
from libact.query_strategies.density_weighted_uncertainty_sampling import DWUS, DensityWeightedLogisticRegression
from libact.base.dataset import Dataset
from libact.models import SklearnProbaAdapter
from cardinAL.base import BaseQuerySampler
from sklearn.model_selection import train_test_split


from libact.base.interfaces import QueryStrategy


class LibactNoSampling(QueryStrategy):
    def __init__(self, dataset, **kwargs):
        super(LibactNoSampling, self).__init__(dataset, **kwargs)
        pass

    def _get_scores(self):
        dataset = self.dataset
        #self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()

        return zip(unlabeled_entry_ids, [1] * len(unlabeled_entry_ids))

    def make_query(self, return_score=False):
        unlabeled_entry_ids, scores = zip(*self._get_scores())
        ask_id = np.argmax(scores)

        if return_score:
            return unlabeled_entry_ids[ask_id], \
                   list(zip(unlabeled_entry_ids, scores))
        else:
            return unlabeled_entry_ids[ask_id]


class LibactDensityMeta(BaseQuerySampler):

    def __init__(self, estimator, batch_size):
        super().__init__(batch_size=batch_size)
        self.estimator = estimator

    def fit(self, X, y):

        # Libact object requires to have access to the whole data at init time.
        # UGLY We store it, no other choice.
        self.X_train = X
        self.y_train = y

    def select_samples(self, X):

        # Libact do its selection on the whole dataset. Merging train and test:
        n_train, n_test = self.X_train.shape[0], X.shape[0]
        X = np.vstack([self.X_train, X])
        # y is of shape n_train + n_test, we fill the test samples with None for each class
        y = self.y_train.tolist() + [None] * n_test

        dataset = Dataset(X, y)

        model = DensityWeightedMeta(
            dataset,
            LibactNoSampling(dataset))

        selected_idx = model._get_scores()
        selected_idx = [i - n_train for i, _ in sorted(selected_idx, key = lambda x: x[1])[-self.batch_size:]]

        return selected_idx


class LibactDWUS(BaseQuerySampler):

    def __init__(self, estimator, batch_size):
        super().__init__(batch_size=batch_size)
        self.estimator = estimator

    def fit(self, X, y):

        # Libact object requires to have access to the whole data at init time.
        # UGLY We store it, no other choice.
        self.X_train = X
        self.y_train = y

    def select_samples(self, X):

        # Libact do its selection on the whole dataset. Merging train and test:
        n_train, n_test = self.X_train.shape[0], X.shape[0]
        X = np.vstack([self.X_train, X])
        # y is of shape n_train + n_test, we fill the test samples with None for each class
        y = self.y_train.tolist() + [None] * n_test

        dataset = Dataset(X, y)

        model = DWUS(
            dataset,
            model=LibactNoSampling(dataset))

        unlabeled_entry_ids, _ = model.dataset.get_unlabeled_entries()
        labeled_entry_ids = np.array([eid
                                      for eid, x in enumerate(model.dataset.data)
                                      if x[1] is not None])
        labels = np.array([x[1]
                           for eid, x in enumerate(model.dataset.data)
                           if x[1] is not None]).reshape(-1, 1)
        centers = model.kmeans_.cluster_centers_
        P_k_x = model.P_k_x
        p_x = model.p_x[list(unlabeled_entry_ids)]

        clf = DensityWeightedLogisticRegression(P_k_x[labeled_entry_ids, :],
                                                centers,
                                                model.C)
        clf.train(labeled_entry_ids, labels)
        P_y_k = clf.predict()

        P_y_x = np.zeros(len(unlabeled_entry_ids))
        for k, center in enumerate(centers):
            P_y_x += P_y_k[k] * P_k_x[unlabeled_entry_ids, k]

        # binary case
        expected_error = P_y_x
        expected_error[P_y_x >= 0.5] = 1. - P_y_x[P_y_x >= 0.5]

        scores = expected_error * p_x

        selected_idx = enumerate(scores)
        selected_idx = [i for i, _ in sorted(selected_idx, key = lambda x: x[1])[-self.batch_size:]]

        return selected_idx


class ModalDensity():

    def __init__(self, n_samples, metric='euclidean'):
        self.n_samples = n_samples
        self.metric = metric

    def fit(self, X, y):
        return self
    
    def select_samples(self, X):
        density = information_density(X, self.metric)
        self.scores_ = density
        return np.argsort(density)[-self.n_samples:]


results = dict()

model = SVC(gamma='scale')

n_samples = 1000
n_iter = 20
batch_size = n_samples // n_iter // 2
n_classes = 5

alipy = AlipyWrapper(model, QueryInstanceGraphDensity, batch_size, train_idx=True)

libact = LibactDensityMeta(SVC, batch_size)
libact2 = LibactDWUS(SVC, batch_size)

modal_euclidean = ModalDensity(batch_size)
modal_cosine = ModalDensity(batch_size, metric='cosine')

rand = RandomSampler(batch_size)


results = dict()
for name, method in [
        ('alipy', alipy),
        ('libact-KMeans', libact),
        ('libact-GMM', libact2),
        ('modal-euclidean', modal_euclidean),
        ('modal-euclidean-weighted', modal_euclidean),
        ('modal-cosine', modal_cosine),
        ('random', rand)
    ]:
    folds = []
    for i in range(10):
        X, y = make_classification(n_samples=n_samples + n_classes, n_features=15, n_informative=7, n_redundant=0,
                           n_repeated=0, n_classes=n_classes, n_clusters_per_class=10, weights=None, flip_y=0.01, class_sep=10.0,
                           hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=i)
        
        split_index = n_samples // 2 + n_classes
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=i)
        
        selected = np.zeros(X_train.shape[0], dtype=bool)
        index = np.arange(X_train.shape[0])
        weights = np.ones(X_train.shape[0])
        # We randomly selected one sample of each class to avoid cases where the training set has only one class
        for i in range(n_classes):
            selected[np.where(y_train == i)[0][0]] = True
        scores = []

        for j in range(n_iter):
            method.fit(X_train[selected], y_train[selected])
            new_selected = method.select_samples(X_train[~selected])
            selected[index[~selected][new_selected]] = True
            if name == 'modal-euclidean-weighted':
                weights[new_selected] = 1. / (method.scores_[new_selected])
                print(weights)
                score = model.fit(X_train[selected], y_train[selected], weights[selected]).score(X_test, y_test)
            else:
                score = model.fit(X_train[selected], y_train[selected]).score(X_test, y_test)
            scores.append(score)
        folds.append(scores)
    folds = np.asarray(folds)
    results[name] = folds

    plot_confidence_interval(np.arange(1, n_iter + 1) * 100 // n_iter, folds, label=name, smoothing=10)

plt.xlabel('Percentage of train samples selected')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
