import os
import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy as sp

import cardinal
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, pairwise_distances, silhouette_score, calinski_harabaz_score

from cardinal.uncertainty import MarginSampler, ConfidenceSampler, EntropySampler
from cardinal.random import RandomSampler
from cardinal.clustering import KMeansSampler, KCentroidSampler, MiniBatchKMeansSampler
from cardinal.zhdanov2019 import TwoStepKMeansSampler
from cardinal.batch import RankedBatchSampler

from experimenter import Experiment

suffix = ''

cache_folder = './cache{}'.format(suffix)
database_path = 'sqlite:///database{}.db'.format(suffix)


from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

cv = CountVectorizer(ngram_range=(1,2), max_features=250000)
tf = TfidfTransformer(use_idf=True) 

X_train = cv.fit_transform(train.data)
X_train = tf.fit_transform(X_train)
X_test = cv.transform(test.data)
X_test = tf.transform(X_test)
y_train = train.target
y_test = test.target




X = sp.sparse.vstack([X_train, X_test])
y = np.hstack([y_train, y_test])

mm_dist = pairwise_distances(X, X)

#nrows = ncols = X.shape[0]
#mm_dist = np.memmap('dist_memmapped.dat', dtype=np.float32,
#                    shape=(nrows, ncols))

start_size = 100
batch_size = 100
stop_size = 2000

n_classes = 20 

dataset_name = 'news'

k_start = False
is_keras = False

if is_keras:
    import keras
    from keras.layers import Dropout, Dense, GlobalAveragePooling2D
    from keras import Model
    from keras import optimizers, layers
    from keras.models import Sequential
    from keras.callbacks import EarlyStopping


def get_clf():
    return SGDClassifier(loss='log', class_weight='balanced')

def fit_clf(clf, tx, ty):
    clf.fit(tx, ty)


class TwoStepMiniBatchKMeansSampler(TwoStepKMeansSampler):
    def __init__(self, beta: int, classifier, batch_size: int,
                 assume_fitted: bool = False, verbose: int = 0, **kmeans_args):
        
        self.sampler_list = [
            MarginSampler(classifier, beta * batch_size, strategy='top',
                          assume_fitted=assume_fitted, verbose=verbose),
            MiniBatchKMeansSampler(batch_size, **kmeans_args)
        ]

    def select_samples(self, X: np.array,
                       ) -> np.array:
        selected = self.sampler_list[0].select_samples(X)
        new_selected = self.sampler_list[1].select_samples(
            X[selected], sample_weight=self.sampler_list[0].sample_scores_[selected])
        selected = selected[new_selected]

        return selected


def get_min_dist_per_class(dist, labels):
    assert(dist.shape[0] == labels.shape[0])
    min_dist_per_class = np.zeros((dist.shape[1], n_classes))
    max_dist = np.max(dist)  # If a class is not represented we use this as max distance

    for ic in range(n_classes):
        mask_sample_of_class_ic = dist[labels == ic]
        if mask_sample_of_class_ic.shape[0] != 0:
            min_dist_per_class[:, ic] = np.min(dist[labels == ic], axis=0)
        else :
            min_dist_per_class[:, ic].fill(max_dist)
    
    return min_dist_per_class


for seed, ds in itertools.product(['11', '22', '33', '44', '55'], ['A', 'B']):
    print(seed, ds)
    methods = {
        'random': RandomSampler(batch_size=batch_size, random_state=int(seed)),
        'margin': lambda clf: MarginSampler(clf, batch_size=batch_size, assume_fitted=True),
        'uncertainty': lambda clf: ConfidenceSampler(clf, batch_size=batch_size, assume_fitted=True),
        'entropy': lambda clf: EntropySampler(clf, batch_size=batch_size, assume_fitted=True),
        'kmeans': KCentroidSampler(MiniBatchKMeans(n_clusters=batch_size, n_init=1, random_state=int(seed)), batch_size=batch_size),
        'wkmeans': lambda clf: TwoStepMiniBatchKMeansSampler(10, clf, batch_size, assume_fitted=True, n_init=1, random_state=int(seed)),
    }
    
    index = np.arange(X.shape[0])

    init_train, init_test = train_test_split(index, test_size=.5, random_state=int(seed))
    if ds == 'B':
        init_train, init_test = init_test, init_train
    
    for name in methods:
        print(name)
        method = methods[name]
        exp = Experiment(database_path, int(seed), folder=os.path.join(cache_folder, name, ds))
        index_train = exp.variable('index_train', init_train)
        index_test = exp.variable('index_test', init_test)
        X_train, X_test, y_train, y_test = X[index_train.val], X[index_test.val], y[index_train.val], y[index_test.val]
        
        if len(y_test.shape) == 2:
            y_test = np.argmax(y_test, axis=1)

        index = np.arange(X_train.shape[0])
        first_selected = np.zeros(X_train.shape[0], dtype=bool)

        if not k_start:
            first_index, _ = train_test_split(index, train_size=start_size, random_state=int(seed), stratify=y_train)
            first_selected[first_index] = True
        else:
            start_sampler = MiniBatchKMeansSampler(start_size, random_state=int(seed))
            start_sampler.fit(X_train)
            first_index = start_sampler.select_samples(X_train)
            first_selected[first_index] = True
        
        classifier = exp.variable('clf', get_clf())
        
        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_selected'):
            new_selected = exp.variable('new_selected', first_selected)
            selected = exp.variable('selected', new_selected.val)
            selected.val = new_selected.val.copy()
            
            fit_clf(classifier.val, X_train[selected.val], y_train[selected.val])

            predicted = exp.variable('predicted', None)
            predicted.val = classifier.val.predict_proba(X_test)
            predicted_train = exp.variable('predicted_train', None)
            predicted_train.val = classifier.val.predict_proba(X_train)
            
            if name in ['uncertainty', 'margin', 'entropy', 'wkmeans']:
                sampler = method(classifier.val)
            else:
                sampler = method
            sampler.fit(X_train[selected.val], y_train[selected.val])
            new_selected_index = sampler.select_samples(X_train[~selected.val])
            mask = new_selected.val
            mask[index[~selected.val][new_selected_index]] = True
            new_selected.val = mask
            print(n_samples, selected.val.sum())

            
        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_accuracies'):
            selected = exp.variable('selected', None)
            predicted = exp.variable('predicted', None)
            
            config = dict(
                seed=seed + ds,
                method=name,
                n_samples=n_samples,
                dataset=dataset_name
            )

            exp.log_value(config, 'accuracy', accuracy_score(y_test, np.argmax(predicted.val, axis=1)))
            print(selected.val.sum())
            print('acc', accuracy_score(y_test, np.argmax(predicted.val, axis=1)))


#        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_cluster_metrics'):
#            selected = exp.variable('selected', None)
#            predicted = exp.variable('predicted', None)
#            
#            config = dict(
#                seed=seed + ds,
#                method=name,
#                n_samples=n_samples,
#                dataset=dataset_name
#            )
#            # dist = pairwise_distances(X_train[selected.val], X_test)
#            dist = mm_dist[index_train.val[selected.val][:, None], index_test.val]
#            assert(dist.shape == (selected.val.sum(), index_test.val.shape[0]))
#            labels = np.argmin(dist, axis=0)
#            sse = (dist[labels, np.arange(X_test.shape[0])] ** 2).sum()
#            assert(index_train.val.shape[0] == labels.shape[0])
#            exp.log_value(config, 'distortion', sse.item())
#            exp.log_value(config, 'calinski', calinski_harabaz_score(X_test, labels))
#            exp.log_value(config, 'silhouette', silhouette_score(X_test, labels).item())
            

        prev_pred = None
        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_contradictions'):
            predicted = exp.variable('predicted', None)

            config = dict(
                seed=seed + ds,
                method=name,
                n_samples=n_samples,
                dataset=dataset_name
            )

            if prev_pred is not None:
                # Hard conrtadictions
                exp.log_value(config, 'hard_contradiction', (np.sum(np.argmax(prev_pred, axis=1) == np.argmax(predicted.val, axis=1)) / y_test.shape[0]).item())
                exp.log_value(config, 'soft_contradiction', (np.sum(np.abs(prev_pred - predicted.val)) / y_test.shape[0]).item())
                exp.log_value(config, 'top_contradiction', (np.sum(np.max(prev_pred, axis=1) - np.max(predicted.val, axis=1)) / y_test.shape[0]).item())
            prev_pred = predicted.val.copy()
            
        prev_dist = None
        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_exploration'):
            selected = exp.variable('selected', None)
            predicted = exp.variable('predicted', None)

            config = dict(
                seed=seed + ds,
                method=name,
                n_samples=n_samples,
                dataset=dataset_name
            )
            # dist = pairwise_distances(X_train[selected.val], X_test)
            dist = mm_dist[index_train.val[selected.val][:, None], index_test.val]

            # One hot encode train labels
            predicted_train = exp.variable('predicted_train', None)
            labels_train = np.argmax(predicted_train.val[selected.val], axis=1)
            
            min_dist_per_class = get_min_dist_per_class(dist, labels_train)
            
            if prev_dist is not None:
                exp.log_value(config, 'hard_exploration', (np.sum(np.argmin(prev_dist, axis=1) == np.argmin(min_dist_per_class, axis=1)) / y_test.shape[0]).item())
                exp.log_value(config, 'soft_exploration', (np.sum(np.abs(prev_dist - min_dist_per_class)) / y_test.shape[0]).item())
                exp.log_value(config, 'top_exploration', (np.sum(np.min(prev_dist, axis=1) - np.min(min_dist_per_class, axis=1)) / y_test.shape[0]).item())
            prev_dist = min_dist_per_class.copy()

        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_trustscores'):
            selected = exp.variable('selected', None)
            predicted = exp.variable('predicted', None)

            config = dict(
                seed=seed + ds,
                method=name,
                n_samples=n_samples,
                dataset=dataset_name
            )
            # dist = pairwise_distances(X_train[selected.val], X_test)
            dist = mm_dist[index_train.val[selected.val][:, None], index_test.val]

            # One hot encode train labels
            predicted_train = exp.variable('predicted_train', None)
            labels_train = y_train[selected.val]
            
            d = get_min_dist_per_class(dist, labels_train)
            labels_test = np.argmax(predicted.val, axis=1)

            sorted_d = np.sort(d, axis=1)
            d_to_pred = d[range(d.shape[0]), labels_test]
            d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred,
                                             sorted_d[:, 0], sorted_d[:, 1])
            exp.log_value(config, 'trust_score', np.mean(d_to_closest_not_pred / (d_to_pred + 1e-10)).item()) 

        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_train_accuracies'):
            selected = exp.variable('selected', None)
            predicted_train = exp.variable('predicted_train', None)
            
            config = dict(
                seed=seed + ds,
                method=name,
                n_samples=n_samples,
                dataset=dataset_name
            )

            exp.log_value(config, 'train_accuracy', accuracy_score(y_train[selected.val], np.argmax(predicted_train.val[selected.val], axis=1)))

        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_hausdorff'):
            selected = exp.variable('selected', None)
            predicted = exp.variable('predicted', None)

            config = dict(
                seed=seed + ds,
                method=name,
                n_samples=n_samples,
                dataset='mnist'
            )
            

            hausdorff = None
            labels_test = np.argmax(predicted.val, axis=1)
            labels_train = y_train
            for clazz in range(10):
                dist = mm_dist[index_train.val[selected.val][labels_train[selected.val] == clazz][:, None], index_test.val[labels_test == clazz]]
                # dist = pairwise_distances(X_train[selected.val],
                #                           X_test[labels_test == clazz])
                score = max(np.max(np.min(dist, axis=1)), np.max(np.min(dist, axis=0)))
                if hausdorff is None or score > hausdorff:
                    hausdorff = score
                
            exp.log_value(config, 'hausdorff', hausdorff.item())

        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_batch_trustscores'):
            selected = exp.variable('selected', None)
            new_selected = exp.variable('new_selected', None)

            config = dict(
                seed=seed + ds,
                method=name,
                n_samples=n_samples,
                dataset=dataset_name
            )

            index_batch = np.logical_xor(new_selected.val, selected.val)
            # dist = pairwise_distances(X_train[selected.val], X_test)
            dist = mm_dist[index_train.val[selected.val][:, None], index_train.val[index_batch]]

            # Trust score between batch and labeled
            predicted_train = exp.variable('predicted_train', None)
            labels_train = y_train[selected.val]
            
            d = get_min_dist_per_class(dist, labels_train)

            sorted_d = np.sort(d, axis=1)
            d_to_pred = d[range(d.shape[0]), y_train[index_batch]]
            d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred,
                                             sorted_d[:, 0], sorted_d[:, 1])
            exp.log_value(config, 'batch_trust_score', np.mean(d_to_closest_not_pred / (d_to_pred + 1e-10)).item()) 
            
            # Trust score between batch and well / badly labeled
            predicted_train = exp.variable('predicted_train', None)
            labels_train = y_train[selected.val]
            labels_match = (y_train == np.argmax(predicted_train.val, axis=1)).astype(int)
 
            d = get_min_dist_per_class(dist, labels_match[selected.val])

            sorted_d = np.sort(d, axis=1)
            d_to_pred = d[range(d.shape[0]), np.ones(batch_size, dtype=int)]
            d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred,
                                             sorted_d[:, 0], sorted_d[:, 1])
            exp.log_value(config, 'batch_label_score', np.mean(d_to_closest_not_pred / (d_to_pred + 1e-10)).item()) 
        
        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_binary_trust'):
            selected = exp.variable('selected', None)
            new_selected = exp.variable('new_selected', None)

            config = dict(
                seed=seed + ds,
                method=name,
                n_samples=n_samples,
                dataset=dataset_name
            )

            index_batch = np.logical_xor(new_selected.val, selected.val)
            # dist = pairwise_distances(X_train[selected.val], X_test)
            dist = mm_dist[index_train.val[selected.val][:, None], index_train.val[index_batch]]

            # Trust score between batch and labeled
            predicted_train = exp.variable('predicted_train', None)
            labels_train = y_train[selected.val]
            
            labels_match = (y_train == np.argmax(predicted_train.val, axis=1)).astype(int)
 
            d = get_min_dist_per_class(dist, labels_match[selected.val])
            d = np.argmin(d, axis=1)
            exp.log_value(config, 'batch_binary_score', (d.sum() / d.shape[0]).item())

        for n_samples in exp.iter('exp', range(start_size, stop_size + 1, batch_size), 'generate_agreement'):
            selected = exp.variable('selected', None)
            new_selected = exp.variable('new_selected', None)

            config = dict(
                seed=seed + ds,
                method=name,
                n_samples=n_samples,
                dataset=dataset_name
            )


            # Trust score between batch and labeled
            predicted_train = exp.variable('predicted_train', None)
            predicted = exp.variable('predicted', None)
            labels_train = y_train[selected.val]
            
            # Agreement on batch
            
            index_batch = np.logical_xor(new_selected.val, selected.val)
            dist = mm_dist[index_train.val[selected.val][:, None], index_train.val[index_batch]]
            
            d = get_min_dist_per_class(dist, labels_train)
            nn_predicted = np.argmin(d, axis=1)
            clf_predicted = np.argmax(predicted_train.val[index_batch], axis=1)
            exp.log_value(config, 'batch_agreement', ((nn_predicted == clf_predicted).sum() / batch_size).item())


            # Agreement on test
            
            dist = mm_dist[index_train.val[selected.val][:, None], index_test.val]
            
            d = get_min_dist_per_class(dist, labels_train)
            nn_predicted = np.argmin(d, axis=1)
            clf_predicted = np.argmax(predicted.val, axis=1)
            exp.log_value(config, 'test_agreement', ((nn_predicted == clf_predicted).sum() / index_test.val.shape[0]).item())
