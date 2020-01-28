# This example is drawn from the sklearn gallery and adapted for active learning
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools
import json
import os

from sklearn.datasets import fetch_openml
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


seeds = ['1', '12', '42', '69', '81', '111', '421', '666', '7777', '3']
datasets = ['mnist_sklearn']
classifiers = ['mlp_sklearn']
samplers = ['random', 'uncertainty_10', 'uncertainty_20', 'uncertainty_50', 'uncertainty_100', 'uncertainty_200']

all_results = defaultdict(list)

# Active learning parameters
start_size = 100
stop_size = 1000

for seed, dataset, classifier, sampler in itertools.product(seeds, datasets, classifiers, samplers):

    cache_key = 'uncertainty-size_' + ('_'.join([seed, dataset, classifier, sampler]))
    print('Computing {}'.format(cache_key))

    if os.path.exists(cache_key):
        all_results[(dataset, classifier, sampler)].append(json.load(open(cache_key, 'r')))
        continue

    random_state = check_random_state(int(seed))

    # Datasets
    # Outputs X_train, y_train, X_test, y_test

    if dataset == 'mnist_sklearn':
        # Load data from https://www.openml.org/d/554
        # 70000 samples
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
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

    sampler_elts = sampler.split('_')

    methods = {
        'random': RandomSampler(batch_size=100, random_state=42),
        'uncertainty': UncertaintySampler(clf, batch_size=100),
    }


    if len(sampler_elts) > 1:
        al_model = UncertaintySampler(clf, batch_size=int(sampler_elts[1]))
    else:
        al_model = methods[sampler_elts[0]]

    results = []
    selected = np.zeros(X_train.shape[0], dtype=bool)
    selected[:start_size] = True
    while np.sum(selected) < stop_size:
        
        # Score the model
        clf.fit(X_train[selected], y_train[selected], **fit_params)
        score = clf.score(X_test, y_test)
        results.append((int(np.sum(selected)), score))

        # Select next samples
        if np.sum(selected) != stop_size:
            al_model.fit(X_train[selected], y_train[selected])
            new_selected = al_model.predict(X_train[~selected])
            new_selected = al_model.inverse_transform(new_selected) == 1
            selected[~selected] = new_selected

    all_results[(dataset, classifier, sampler)].append(results)
    json.dump(results, open(cache_key, 'w'))
    # plt.plot(results, label=method)

print('Ended. Plotting...')

for dataset, classifier in itertools.product(datasets, classifiers):
    plt.figure(figsize=(10, 8))
    for sampler in samplers:
        # This is a sequence of results with different seeds, we stack
        data = all_results[(dataset, classifier, sampler)]
        x_labels = [i[0] for i in data[0]]
        scores = [[i[1] for i in j] for j in data]
        data = np.vstack(scores)

        # We extract stats
        avg = np.mean(data, axis=0)
        q10 = np.quantile(data, 0.1, axis=0)
        q90 = np.quantile(data, 0.9, axis=0)

        # Plot the mean line and get its color
        line = plt.plot(x_labels, avg, label=sampler)
        color = line[0].get_c()

        # Plot confidence intervals
        plt.fill_between(x_labels, q90, q10,
                     color=color, alpha=.3)
        
    plt.legend()
    plt.savefig('uncertainty_' + dataset + '_' + classifier + '.png')
