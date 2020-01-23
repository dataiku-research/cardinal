# This example is drawn from the sklearn gallery and adapted for active learning
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools
import json
import os
import pandas as pd

# from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from cardinAL.uncertainty import MarginSampler, ConfidenceSampler, EntropySampler
from cardinAL.random import RandomSampler
from cardinAL.submodularity import SubmodularSampler
from cardinAL.clustering import KMeansSampler, WKMeansSampler
from cardinAL.batch import RankedBatchSampler
from cardinAL.experimental import DeltaSampler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from muscovy_duck import Bencher, ValueStep, SeedingStep, get_keras_cifar, sampler_step, random_sampler_step
from cardinAL.base import ChainQuerySampler
from copy import deepcopy
import keras
from keras.applications.resnet_v2 import ResNet50V2
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras import Model
from keras import optimizers, layers
from keras.models import Sequential
import tensorflow as tf


bencher = Bencher([
    ('get_dataset', True),
    ('config', False),
    ('seeding', False),
    ('create_model', False),
    ('create_sampler', False),
    ('run_experiment', True)], 'zhdanov_cifar')


bencher.register_step('config', '1000-1000-10000', ValueStep(dict(batch_size=1000, start_size=1000, stop_size=10000)))

for seed in ['1', '12', '42', '69', '81', '111', '421', '666', '7777', '3']:
    bencher.register_step('seeding', seed, SeedingStep(int(seed)))

bencher.register_step('get_dataset', 'keras-cifar', get_keras_cifar)

def create_resnet_cifar(data):
    resnet = ResNet50V2(include_top=False, weights='imagenet', input_shape=(200, 200, 3))
    output = resnet.layers[-1].output
    output = keras.layers.Flatten()(output)

    resnet = Model(resnet.input, output=output)

    for layer in resnet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(layers.Lambda(lambda image: tf.image.resize(image, (200, 200))))
    model.add(resnet)
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])

    return {'clf': model}

bencher.register_step('create_model', 'keras-resenet50V2-pretrained', create_resnet_cifar)
#bencher.register_step('create_model', 'keras-resenet50V2', ValueStep(dict(clf=MLPClassifier(hidden_layer_sizes=(128, 64)))))

bencher.register_step('create_sampler', 'random', random_sampler_step)
bencher.register_step('create_sampler', 'margin', lambda data: dict(sampler=MarginSampler(data['clf'], batch_size=data['batch_size'], refit=False)))
#bencher.register_step('create_sampler', 'uncertainty', lambda data: dict(sampler=ConfidenceSampler(data['clf'], batch_size=data['batch_size'])))
#bencher.register_step('create_sampler', 'entropy', lambda data: dict(sampler=EntropySampler(data['clf'], batch_size=data['batch_size'])))

#bencher.register_step('create_sampler', 'kmeans_10', lambda data: dict(sampler=ChainQuerySampler(
#        MarginSampler(data['clf'], batch_size=data['batch_size'] * 10),
#        KMeansSampler(batch_size=data['batch_size'], random_state=data['random_state'])
#    )))
# bencher.register_step('create_sampler', 'wkmeans_10', lambda data: dict(sampler=WKMeansSampler(data['clf'], beta=10, batch_size=data['batch_size'], random_state=data['random_state'])))
# 
# bencher.register_step('create_sampler', 'rankedbatch', lambda data: dict(sampler=RankedBatchSampler(MarginSampler(data['clf'], batch_size=data['batch_size']), alpha='auto', batch_size=data['batch_size'])))
# bencher.register_step('create_sampler', 'delta', lambda data: dict(sampler=DeltaSampler(data['clf'], batch_size=data['batch_size'], n_last=5)))


def plot_results(data):
    # for dataset, classifier in itertools.product(datasets, classifiers):
    # 

    index = pd.DataFrame(data.keys(), columns=bencher.steps)
    accuracies = [v['test_scores'] for v in data.values()]

    # I go with multi index. If you are not familiar with that, just reset the index after
    df = pd.DataFrame(accuracies, index=pd.MultiIndex.from_frame(index))

    x_labels = df.columns.values

    df = df.groupby('create_sampler').agg([
        ('mean', np.mean),
        ('q10', lambda x: np.quantile(x, 0.1, axis=0)),
        ('q90', lambda x: np.quantile(x, 0.9, axis=0))
    ])       

    # Swap the levels to acces directly mean, q10, q90
    df.columns = df.columns.swaplevel()

    plt.figure(figsize=(10, 8))
    for sampler in df.index.values:
        mean = df.loc[sampler]['mean'].values
        q10 = df.loc[sampler]['q10'].values
        q90 = df.loc[sampler]['q90'].values

        # Plot the mean line and get its color
        line = plt.plot(x_labels, mean, label=sampler)
        color = line[0].get_c()

        # Plot confidence intervals
        plt.fill_between(x_labels, q90, q10,
                         color=color, alpha=.3)
        
    plt.legend()
    plt.xlabel('Training sample count')
    plt.ylabel('Accuracy')
    plt.savefig('zhdanov_accuracies.png')



def run_experiment(data):

    random_state = data['random_state']
    already_split = not 'X' in data

    new_data = dict()

    if not already_split:
        # Data is not separated in train and test, we do it
        X = data['X']
        y = data['y']
        X = X.reshape((X.shape[0], -1))

        # Shake the data
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        X = X.reshape((X.shape[0], -1))
        new_data['permutation'] = permutation

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=data['stop_size'], test_size=X.shape[0] - data['stop_size'], random_state=random_state)
    else:
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    start_size = data['start_size']
    stop_size = data['stop_size']
    batch_size = data['batch_size']

    clf = data['clf']

    sampler = data['sampler']

    train_scores = []
    test_scores = []
    train_losses = []
    test_losses = []
    train_embeds = bencher.get_lazy_list('train_embeds')
    test_embeds = bencher.get_lazy_list('test_embeds')
    selecteds = []
    probas_train = []
    probas_test = []
    clfs = []

    selected = np.zeros(X_train.shape[0], dtype=bool)
    selected[:start_size] = True
    random_state.shuffle(selected)
    for n_samples in range(start_size, stop_size + 1, batch_size):
        selecteds.append(selected.copy())

        # Score the model
        clf.fit(X_train[selected], y_train[selected], epochs=3)
        
        # Computing embeddings
        E_model = Model(clf.input, clf.layers[-4].output)
        E_test = E_model.predict(X_test)
        E_train = E_model.predict(X_train)
        test_embeds.save_append(E_test)
        train_embeds.save_append(E_train)

        end_input = layers.Input(clf.layers[-3].input_shape[1:])
        i_layer = end_input
        for layer in clf.layers[-3:]:
            i_layer = layer(i_layer)
        end_model = Model(inputs=end_input, outputs=i_layer)
        end_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])

        # Gather metrics
        test_loss, test_score = end_model.evaluate(E_test, y_test, batch_size=128)
        test_scores.append(test_score)
        test_losses.append(test_loss)
        
        train_loss, train_score = end_model.evaluate(E_train, y_train, batch_size=128)
        train_scores.append(train_score)
        train_losses.append(train_loss)

        probas_train.append(end_model.predict(E_train, batch_size=128, verbose=1))
        probas_test.append(end_model.predict(E_test, batch_size=128, verbose=1))
        clfs.append(deepcopy(end_model))

        del E_test

        # Select next samples
        if n_samples != stop_size:
            if hasattr(sampler, 'classifier_'):
                sampler.classifier_ = end_model
            sampler.fit(E_train[selected], y_train[selected])
            new_selected = sampler.predict(E_train[~selected])
            print(new_selected.sum())
            selected[~selected] = new_selected
        del E_train

    new_data['train_scores'] = train_scores
    new_data['test_scores'] = test_scores
    new_data['train_losses'] = train_losses
    new_data['test_losses'] = test_losses
    new_data['train_embeds'] = train_embeds
    new_data['test_embeds'] = test_embeds
    new_data['selected'] = selecteds
    new_data['probas_train'] = probas_train
    new_data['probas_test'] = probas_test
    new_data['clfs'] = clfs

    return new_data
    

bencher.register_step('run_experiment', 'simple_exp', run_experiment)

bencher.register_reducer('accuracies', plot_results)

bencher.run()

# print('Ended. Plotting...')
# 
