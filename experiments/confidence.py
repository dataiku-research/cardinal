from cardinAL.alipy import AlipyWrapper
from cardinAL.libact import LibactWrapper
from cardinAL.modal import ModalWrapper
from alipy.query_strategy.query_labels import QueryInstanceUncertainty
from libact.query_strategies.uncertainty_sampling import UncertaintySampling
from modAL.uncertainty import uncertainty_sampling
from modAL.models import ActiveLearner
# This example is drawn from the sklearn gallery and adapted for active learning
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools
import json
import os
import timeit

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from cardinAL.uncertainty import UncertaintySampler
from cardinAL.random import RandomSampler
from cardinAL.submodularity import SubmodularSampler
from cardinAL.clustering import KMeansSampler, WKMeansSampler
from cardinAL.batch import RankedBatchSampler
from cardinAL.experimental import DeltaSampler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


class Cache():

    def __init__(self, dir_):
        self.dir = dir_

    def fp(self, cache_key):
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        return os.path.join(self.dir, cache_key)


seeds = ['1', '12', '42', '69', '81', '111', '421', '666', '7777', '3']
datasets = ['mnist_sklearn']
classifiers = ['mlp_sklearn']
samplers = ['random', 'cardinal', 'modal', 'alipy', 'libact']

all_results = defaultdict(list)

# Active learning parameters
batch_size = 100
start_size = 100
stop_size = 1000

exp_name = 'confidence'

cache = Cache(exp_name)

for seed, dataset, classifier, sampler in itertools.product(seeds, datasets, classifiers, samplers):

    cache_key = '_'.join([seed, dataset, classifier, sampler])
    print('Computing {}'.format(cache_key))

    if os.path.exists(cache.fp(cache_key)):
        all_results[(dataset, classifier, sampler)].append(json.load(open(cache.fp(cache_key), 'r')))
        continue

    random_state = check_random_state(int(seed))

    # Datasets
    # Outputs X_train, y_train, X_test, y_test

    if dataset == 'mnist_sklearn':
        # Requires latest version of sklearn. We load it from a file if not available
        # Load data from https://www.openml.org/d/554
        # 70000 samples

        try:
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        except ImportError:
            if not (os.path.exists('mnist_784_X.npy') and os.path.exists('mnist_784_y.npy')):
                raise ValueError('Dataset not available')
            X = np.load('mnist_784_X.npy')
            y = np.load('mnist_784_y.npy', allow_pickle=True)

        # Shake the data
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        X = X.reshape((X.shape[0], -1))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=stop_size, test_size=X.shape[0] - stop_size)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


    # Classifier
    # Outputs: clf, fit_params
    fit_params = {}

    if classifier == 'mlp_sklearn':
        clf = MLPClassifier(hidden_layer_sizes=(128, 64))

    # Sampling
    # Outputs: al_model

    methods = {
        'random': RandomSampler(batch_size=batch_size, random_state=42),
        'modal': ModalWrapper(ActiveLearner(clf, uncertainty_sampling), batch_size=batch_size, refit=False),
        'alipy': AlipyWrapper(clf, lambda x,y: QueryInstanceUncertainty(x, y, measure='least_confident'), batch_size=batch_size),
        'libact': LibactWrapper(clf, UncertaintySampling, batch_size=batch_size),
	    'cardinal': UncertaintySampler(clf, batch_size=batch_size)
    }

    al_model = methods[sampler]

    # range(start_size, stop_size + 1, batch_size)

    results = []
    selected = np.zeros(X_train.shape[0], dtype=bool)
    selected[:start_size] = True
    tts = 0.
    for n_samples in range(start_size, stop_size + 1, batch_size):
        
        # Score the model
        clf.fit(X_train[selected], y_train[selected], **fit_params)
        score = clf.score(X_test, y_test)
        results.append(score)

        # Select next samples
        if n_samples != stop_size:
            tts += timeit.timeit(lambda: al_model.fit(X_train[selected], y_train[selected]), number=1)
            new_selected = al_model.predict(X_train[~selected])
            new_selected = al_model.inverse_transform(new_selected) == 1
            selected[~selected] = new_selected

    all_results[(dataset, classifier, sampler)].append([tts, results])
    json.dump([tts, results], open(cache.fp(cache_key), 'w'))


print('Ended. Plotting...')
mapper = {
    'random': 'Random',
    'modal': 'modAL',
    'libact': 'libact',
    'alipy': 'Alipy',
    'alipy-graph': 'Alipy',
    'cardinal': 'cardinAL'
}

x_labels = list(range(start_size, stop_size + 1, batch_size))

for dataset, classifier in itertools.product(datasets, classifiers):
    plt.figure(figsize=(10, 8))
    for sampler in samplers:
        # This is a sequence of results with different seeds, we stack
        results = all_results[(dataset, classifier, sampler)]
        tts, results = zip(*results)
        data = np.vstack(results)
        print(sampler, tts)

        # We extract stats
        avg = np.mean(data, axis=0)
        q10 = np.quantile(data, 0.1, axis=0)
        q90 = np.quantile(data, 0.9, axis=0)

        # Plot the mean line and get its color
        line = plt.plot(x_labels, avg, label=mapper[sampler])
        color = line[0].get_c()

        # Plot confidence intervals
        plt.fill_between(x_labels, q90, q10,
                     color=color, alpha=.3)
        
    plt.legend()
    plt.xlabel('Number of MNIST sample labeled')
    plt.ylabel('Accuracy')
    plt.savefig(exp_name + '_' + dataset + '_' + classifier + '.png')
