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
from cardinAL.batch import RankedBatchSampler
from cardinAL.uncertainty import ConfidenceSampler

from alipy.query_strategy.query_labels import QueryInstanceBMDR
from libact.query_strategies import DensityWeightedMeta


results = dict()

model = SVC(gamma='scale', probability=True)

n_samples = 500
n_iter = 20
batch_size = n_samples // n_iter // 2
n_classes = 2

alipy = AlipyWrapper(model, QueryInstanceBMDR, batch_size, train_idx=True)
libact = LibactWrapper(SVC, DensityWeightedMeta, batch_size)

modal = RankedBatchSampler(ConfidenceSampler(model, batch_size), batch_size)

rand = RandomSampler(batch_size)


results = dict()
for name, method in [
        ('alipy', alipy),
        # ('libact', libact),
        ('modal', modal),
        ('random', rand)
    ]:
    folds = []
    for i in range(10):
        X, y = make_classification(n_samples=n_samples + n_classes, n_features=15, n_informative=7, n_redundant=0,
                           n_repeated=0, n_classes=n_classes, n_clusters_per_class=10, weights=None, flip_y=0.01, class_sep=10.0,
                           hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=i)
        
        split_index = n_samples // 2 + n_classes
        
        X_train = X[:split_index]
        y_train = y[:split_index]
        X_test = X[split_index:]
        y_test = y[split_index:]
        
        selected = np.zeros(X_train.shape[0], dtype=bool)
        # We randomly selected one sample of each class to avoid cases where the training set has only one class
        for i in range(n_classes):
            selected[np.where(y_train == i)[0][0]] = True
        scores = []
        for j in range(n_iter):
            method.fit(X_train[selected], y_train[selected])
            new_selected = method.predict(X_train[~selected])
            selected[~selected] = new_selected
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
