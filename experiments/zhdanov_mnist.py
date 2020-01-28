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
from cardinAL.base import UncertaintySampler, RandomSampler, SubmodularSampler, KMeansSampler, WKMeansSampler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


class Cache():
    pass

# seeds = ['1', '12', '42', '69', '81', '111', '421', '666', '7777', '3']
# datasets = ['mnist_sklearn', 'mnist_mxnet']
# classifiers = ['mlp_sklearn', 'mlp_mxnet']
# samplers = ['random', 'uncertainty', 'submodular_10', 'kmeans_10', 'kmeans_50', 'wkmeans_50']

seeds = ['1', '12', '42', '69', '81', '111', '421', '666', '7777', '3']
datasets = ['mnist_mxnet']
classifiers = ['mlp_sklearn', 'mlp_keras']
samplers = ['random', 'uncertainty', 'submodular_10', 'kmeans_10', 'kmeans_50', 'wkmeans50']

all_results = defaultdict(list)

# Active learning parameters
batch_size = 100
start_size = 100
stop_size = 1000

for seed, dataset, classifier, sampler in itertools.product(seeds, datasets, classifiers, samplers):

    cache_key = '_'.join([seed, dataset, classifier, sampler])
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
    
    elif dataset == 'mnist_mxnet':

        import mxnet as mx
        from mxnet import gluon
        
        train_data = mx.gluon.data.DataLoader(
            mx.gluon.data.vision.MNIST(train=True, transform=lambda data, label: (data.astype(np.float32)/255, label)),
            batch_size=np.inf, shuffle=False)
        X_train, y_train = list(train_data)[0]
        X_train = X_train.reshape(X_train.shape[0], -1)
        permutation = random_state.permutation(X_train.shape[0])
        X_train = X_train[permutation].asnumpy()
        y_train = y_train[permutation].asnumpy()

        test_data = mx.gluon.data.DataLoader(
            mx.gluon.data.vision.MNIST(train=False, transform=lambda data, label: (data.astype(np.float32)/255, label)),
            batch_size=np.inf, shuffle=False)

        X_test, y_test = list(train_data)[0]
        X_test = X_test.reshape(X_test.shape[0], -1).asnumpy()
        y_test = y_test.asnumpy()

    # Classifier
    # Outputs: clf, fit_params
    fit_params = {}

    if classifier == 'mlp_sklearn':
        clf = MLPClassifier(hidden_layer_sizes=(128, 64))
    elif classifier == 'mlp_keras':

        from keras.wrappers.scikit_learn import KerasClassifier
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras import optimizers
        from keras.callbacks import EarlyStopping

        callback = EarlyStopping(monitor='loss', min_delta=1e-5, patience = 5)

        def make_model():
            model = Sequential()
            model.add(Dense(128, activation='relu', input_shape=(784,)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.SGD(lr=0.01),
                          metrics=['accuracy'])
            return model
        clf = KerasClassifier(make_model, batch_size=32)
        fit_params['epochs'] = 2000
        fit_params['callbacks'] = [callback]

    else:
        # Initialize the model
        clf = gluon.nn.Sequential()

	# Define the model architecture
        with net.name_scope():
            clf.add(gluon.nn.Dense(128, activation="relu")) # 1st layer - 128 nodes
            clf.add(gluon.nn.Dense(64, activation="relu")) # 2nd layer – 64 nodes
            clf.add(gluon.nn.Dense(10)) # Output layer, one for each number 0-9

    # Sampling
    # Outputs: al_model

    sampler_elts = sampler.split('_')

    methods = {
        'random': RandomSampler(batch_size=batch_size, random_state=42),
        'uncertainty': UncertaintySampler(clf, batch_size=batch_size, random_state=42),
        'submodular': SubmodularSampler(batch_size=batch_size, random_state=42),
        'kmeans': KMeansSampler(batch_size=batch_size, random_state=42),
        'wkmeans50': WKMeansSampler(clf, beta=50, batch_size=batch_size, random_state=42)
    }

    al_model = methods[sampler_elts[0]]

    if len(sampler_elts) > 1:
        # This is a preselection step. We take batch_size * beta samples using uncertainty sampling
        # and then use the other sampler on top of it.
        al_model = Pipeline([
            ('uncertainty', UncertaintySampler(clf, batch_size=batch_size * int(sampler_elts[1]), random_state=42)),
            ('al', al_model)])


    # range(start_size, stop_size + 1, batch_size)

    results = []
    selected = np.zeros(X_train.shape[0], dtype=bool)
    selected[:start_size] = True
    for n_samples in range(start_size, stop_size + 1, batch_size):
        
        # Score the model
        clf.fit(X_train[selected], y_train[selected], **fit_params)
        score = clf.score(X_test, y_test)
        results.append(score)

        # Select next samples
        if n_samples != stop_size:
            al_model.fit(X_train[selected], y_train[selected])
            new_selected = al_model.predict(X_train[~selected])
            new_selected = al_model.inverse_transform(new_selected) == 1
            selected[~selected] = new_selected

    all_results[(dataset, classifier, sampler)].append(results)
    json.dump(results, open(cache_key, 'w'))
    # plt.plot(results, label=method)

print('Ended. Plotting...')

x_labels = list(range(start_size, stop_size + 1, batch_size))

for dataset, classifier in itertools.product(datasets, classifiers):
    plt.figure(figsize=(10, 8))
    for sampler in samplers:
        # This is a sequence of results with different seeds, we stack
        data = np.vstack(all_results[(dataset, classifier, sampler)])

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
    plt.savefig(dataset + '_' + classifier + '.png')
