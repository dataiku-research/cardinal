# This example is drawn from the sklearn gallery and adapted for active learning
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools
import json
import os

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from cardinAL.uncertainty import MarginSampler, EntropySampler, margin_sampling
from cardinAL.random import RandomSampler
from cardinAL.submodularity import SubmodularSampler
from cardinAL.clustering import KMeansSampler, WKMeansSampler
from cardinAL.batch import RankedBatchSampler
from cardinAL.experimental import DeltaSampler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances
from sklearn.semi_supervised import LabelSpreading
from scipy.sparse import vstack


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
from cardinAL.base import BaseQuerySampler
from cardinAL.uncertainty import MarginSampler, ConfidenceSampler, EntropySampler
from cardinAL.random import RandomSampler
from cardinAL.submodularity import SubmodularSampler
from cardinAL.clustering import KMeansSampler, WKMeansSampler
from cardinAL.batch import RankedBatchSampler
from cardinAL.experimental import DeltaSampler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from muscovy_duck import Bencher, ValueStep, SeedingStep, get_keras_cifar, sampler_step, random_sampler_step
from cardinAL.base import ChainQuerySampler
from copy import deepcopy


bencher = Bencher([
    ('get_dataset', True),
    ('config', False),
    ('seeding', False),
    ('create_model', False),
    ('create_sampler', False),
    ('run_experiment', True)], 'zhdanov_news')


bencher.register_step('config', '100-100-1000', ValueStep(dict(batch_size=100, start_size=100, stop_size=1000)))

for seed in ['1', '12', '42', '69', '81', '111', '421', '666', '7777', '3']:
    bencher.register_step('seeding', seed, SeedingStep(int(seed)))

def get_20_news(data):
    from sklearn.datasets import fetch_20newsgroups 
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
    
    cv = CountVectorizer(ngram_range=(1,2), max_features=250000)
    tf = TfidfTransformer(use_idf=True) 

    X_train = cv.fit_transform(train.data)
    X_train = tf.fit_transform(X_train)
    y_train = train.target
    X_test = cv.transform(test.data)
    X_test = tf.transform(X_test)
    y_test = test.target

    # dist_X = pairwise_distances(X_train)

    data.update({
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        # "X_train_dist": dist_X
    })
    return data

bencher.register_step('get_dataset', '20news', get_20_news)

def create_sklearn_log(data):
    data['clf'] = SGDClassifier(loss='log', class_weight='balanced')
    return data

bencher.register_step('create_model', 'sklearn-log', create_sklearn_log)



class MarginSSLSampler(BaseQuerySampler):
    """Selects samples with greatest confusion between the top two classes.

    Smallest margin sampling uses the difference of predicted probability between
    the top two classes to select the samples on which the model is hesitating
    the most, hence the lowest difference.

    Parameters:
        classifier (sklearn.BaseEstimator): Classifier used to
            determine the prediction confidence. The object must
            comply with scikit-learn interface and expose a
            `predict_proba` method.
        batch_size (int): Number of samples to draw when predicting.
        assume_fitted (bool): If true, classifier is not refit
        verbose (int, optional): The verbosity level. Defaults to 0.
    
    Attributes:
        classifier_ (sklearn.BaseEstimator): The fitted classifier.
    """

    def __init__(self, classifier, batch_size, assume_fitted=False, verbose=0):
        super().__init__()
        # TODO: can we check that the classifier has a predict_proba?
        self.classifier_ = classifier
        self.batch_size = batch_size
        self.assume_fitted = assume_fitted
        self.verbose = verbose
        if self.classifier_ == 'precomputed':
            self.assume_fitted = True

    def fit(self, X, y):
        """Fit the estimator on labeled samples.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Training data
            y (numpy array, shape (n_samples,)): Target values

        Returns:
            self: An instance of self.
        """
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        """Selects the samples to annotate from unlabelled data.

        Args:
            X ({array-like, sparse matrix}, shape (n_samples, n_features)): Samples to evaluate.

        Returns:
            predictions (np.array): Returns an array where selected samples are classified as 1.
        """

        # Train clf
        n_train, n_test = self.X_train.shape[0], X.shape[0]
        X_ = vstack([self.X_train, X])
        # y is of shape n_train + n_test, we fill the test samples with 0 for each class
        y_ = np.hstack([self.y_train, -np.ones(n_test)])
        
        lp_model = LabelSpreading(gamma=.25, max_iter=20)
        lp_model.fit(X_, y_)
        predicted_labels = lp_model.transduction_
        self.classifier_.fit(X_, predicted_labels)

        selected_samples = np.zeros(X.shape[0])
        index, confidence = margin_sampling(self.classifier_, X, n_instances=X.shape[0])
        
        self.confidence_ = confidence
        index = index[:self.batch_size]
        
        selected_samples[index] = 1
        self.labels_ = selected_samples

        return selected_samples


bencher.register_step('create_sampler', 'random', random_sampler_step)
bencher.register_step('create_sampler', 'margin', lambda data: dict(sampler=MarginSampler(data['clf'], batch_size=data['batch_size'], assume_fitted=True)))
bencher.register_step('create_sampler', 'margin_ssl', lambda data: dict(sampler=MarginSSLSampler(data['clf'], batch_size=data['batch_size'], assume_fitted=True)))
#bencher.register_step('create_sampler', 'uncertainty', lambda data: dict(sampler=ConfidenceSampler(data['clf'], batch_size=data['batch_size'])))
#bencher.register_step('create_sampler', 'entropy', lambda data: dict(sampler=EntropySampler(data['clf'], batch_size=data['batch_size'])))

bencher.register_step('create_sampler', 'kmeans_10', lambda data: dict(sampler=ChainQuerySampler(
        MarginSampler(data['clf'], batch_size=data['batch_size'] * 10),
        KMeansSampler(batch_size=data['batch_size'], random_state=data['random_state'])
    )))
# bencher.register_step('create_sampler', 'wkmeans_10', lambda data: dict(sampler=WKMeansSampler(data['clf'], beta=10, batch_size=data['batch_size'], random_state=data['random_state'])))
# 
# bencher.register_step('create_sampler', 'rankedbatch', lambda data: dict(sampler=RankedBatchSampler(MarginSampler(data['clf'], batch_size=data['batch_size']), alpha='auto', batch_size=data['batch_size'])))
# bencher.register_step('create_sampler', 'delta', lambda data: dict(sampler=DeltaSampler(data['clf'], batch_size=data['batch_size'], n_last=5)))


def plot_results(data):
    # for dataset, classifier in itertools.product(datasets, classifiers):
    # 

    index = pd.DataFrame(data.keys(), columns=bencher.steps)
    accuracies = [v['accuracies'] for v in data.values()]

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

    start_size = data['start_size']
    stop_size = data['stop_size']
    batch_size = data['batch_size']

    clf = data['clf']

    sampler = data['sampler']

    train_scores = []
    test_scores = []
    train_losses = []
    test_losses = []
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
        clf.fit(X_train[selected], y_train[selected])

        # Gather metrics
        y_test_pred_proba = clf.predict_proba(X_test)
        y_test_pred = clf.classes_[np.argmax(y_test_pred_proba, axis=1)]
        test_scores.append(accuracy_score(y_test, y_test_pred))
        test_losses.append(log_loss(y_test, y_test_pred_proba, labels=clf.classes_))
        
        y_train_pred_proba = clf.predict_proba(X_train)
        y_train_pred = clf.classes_[np.argmax(y_train_pred_proba, axis=1)]
        train_scores.append(accuracy_score(y_train, y_train_pred))
        train_losses.append(log_loss(y_train, y_train_pred_proba, labels=clf.classes_))

        probas_train.append(y_train_pred_proba)
        probas_test.append(y_test_pred_proba)
        clfs.append(deepcopy(clf))

        # Select next samples
        if n_samples != stop_size:
            if hasattr(sampler, 'classifier_'):
                sampler.classifier_ = clf
            sampler.fit(X_train[selected], y_train[selected])
            new_selected = sampler.predict(X_train[~selected])
            print(new_selected.sum())
            selected[~selected] = new_selected

    new_data['train_scores'] = train_scores
    new_data['test_scores'] = test_scores
    new_data['train_losses'] = train_losses
    new_data['test_losses'] = test_losses
    new_data['selected'] = selecteds
    new_data['probas_train'] = probas_train
    new_data['probas_test'] = probas_test
    new_data['clfs'] = clfs

    return new_data
    

bencher.register_step('run_experiment', 'simple_exp', run_experiment)

bencher.register_reducer('accuracies', plot_results)

bencher.run()