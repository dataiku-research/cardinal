"""
Incremental KMeans
==================

In an active learning setting, the trade-off between exploration and
exploitation plays a central role. Exploration, or diversity, is
usually enforced using coresets or, more simply, a clustering algorithm.
KMeans is therefore used to select samples that are spread across the
dataset in each batch. However, to the best of our knowledge, maintaining diversity
throughout the whole experiment is something that has not been considered.

By allowing to start the optimization with fixed cluster centers, our
Incremental KMeans allows for an optimal exploration of the space since
samples are less likely to be selected nearby an already selected sample.

This notebook introduces the principle behind Incremental KMeans and runs
a small benchmark on generated data.

"""

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from cardinal.kmeans import IncrementalMiniBatchKMeans
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
import numpy as np

##############################################################################
# Let us first define the parameters of our experiment. We will generate a
# dataset from 8 clusters and we will try to keep 4 clusters fixed. We do
# this example in 2 dimensions for visualization purposes.

n_clusters = 8
n_fixed_clusters = 4

X, y, centers = make_blobs(centers=n_clusters, return_centers=True,
                           n_features=2, random_state=2)

##############################################################################
# We define plotting functions. This function allows to plot the data colored
# by clusters, and allows to displays fixed cluster centers in red and regular
# centers in blue. We use it to display the ground truth.

def plot_clustering(label, y_pred, centers, fixed_centers=[], inertia=None):
    colors = ['C{}'.format(i) for i in y_pred]
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.scatter(centers[:, 0], centers[:, 1], c='blue', edgecolors='k',
                marker='P', s=200)
    if len(fixed_centers) > 0:
        plt.scatter(fixed_centers[:, 0], fixed_centers[:, 1],
                    c='red', edgecolors='k', marker='P', s=200)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if inertia:
        label = label + ", intertia={0:0.2f}".format(inertia)
    plt.title(label)
    plt.show()

plot_clustering('Ground truth', y, centers)

##############################################################################
# We run a regular MiniBatchKMeans. KMeans would be more suited for this kind
# of small dataset but we are aiming at using Increment KMeans on large
# datasets so its implementation relies on MiniBatchKMeans.

kmeans = MiniBatchKMeans(n_clusters=8, random_state=2)
kmeans.fit(X)
plot_clustering('KMeans', kmeans.predict(X), kmeans.cluster_centers_,
                inertia=kmeans.inertia_)


##############################################################################
# We now consider that we are aware of 4 of the 8 clusters. We fix them in
# the IncrementalMiniBatchKMeans so that they are strictly enforced.

ikmeans = IncrementalMiniBatchKMeans(n_clusters=8, random_state=2)
ikmeans.fit(X, fixed_cluster_centers=centers[:n_fixed_clusters])
plot_clustering(
    'Incremental KMeans', ikmeans.predict(X),
    centers[n_fixed_clusters:],
    fixed_centers=centers[:n_fixed_clusters],
    inertia=ikmeans.inertia_)

##############################################################################
# In this basic experiment, we see that the clusters fixed in Incremental
# KMeans have stayed fixed. It does not improve nor degrade the inertia on a
# very simple problem. We now want to test it in higher dimensions.
#
# Experiments in higher dimension
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We now want to see the effect of some parameters on the inertia. For this
# purpose, we repeat the same experiment on a larger dataset with higher
# dimensions. We explore only a few aspects of the algorithm but our general
# evaluation function can be used to dig deeper in the algorithm.
#
# The following parameters are accessible:
# * `n_repeat` is the number of repetition of the experiment (to see the
#    result variance).
# * `n_features` is the number of dimensions
# * `n_blobs` is the number of clusters in the ground truth
# * `n_clusters` is the number of clusters KMeans is looking for
# * `n_fixed_clusters` is the number of clusters we keep fixed
# * `reassignement_ratio` is a parameter of the MiniBatchKMeans that
#   controls the ratio of centers of high inertia randomly reassigned
# * `recenter_every` is a relaxation of the cluster fixation. 
#   This parameter allows the clusters to move, and fix them back
#   every $k$ iterations. We expect that in some cases, the
#   fixed cluster could be detrimental to the optimization and thus relaxing
#   this constraint could lead to a better minimum.

def evaluate(n_samples, n_repeat, n_features, n_blobs, n_clusters,
             n_fixed_clusters, reassignment_ratio, recenter_every):
    inertiae = []
    for i in range(n_repeat):
        X, y, centers = make_blobs(
            centers=n_blobs,
            return_centers=True,
            n_features=n_features,
            random_state=i)
        clus = IncrementalMiniBatchKMeans(
            n_clusters=n_clusters,
            reassignment_ratio=reassignment_ratio
        )
        kwargs = dict()
        kwargs['fixed_cluster_centers'] = None
        if n_fixed_clusters > 0:
            kwargs['fixed_cluster_centers'] = centers[:n_fixed_clusters]
        kwargs['recenter_every'] = recenter_every
        clus.fit(X, **kwargs)
        inertiae.append(clus.inertia_)
    return inertiae
 

def plot(xlabel, ylabel, values, inertiae):
    arr = np.asarray(inertiae)
    for i in range(arr.shape[1]):
        plt.scatter(values, arr[:, i], alpha=.2, c='gray')
    plt.plot(values, np.mean(arr, axis=1), c='r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


##############################################################################
# Number of fixed cluster centers
# ===============================
#
# In this experiment, we generate data from 10 clusters and see how the number
# of fixed cluster influence the intertia.

study_value = []
study_inert = []

for i in range(10):
    study_value.append(i)
    inert = evaluate(5000, 100, 20, 10, 10, i, 0.01, None)
    study_inert.append(inert)
    
plot('Number of fixed center clusters /10', 'Mean inertia over 100 runs',
     study_value, study_inert)

##############################################################################
# It may seem surprising that the
# inertia is proportional to the number of fixed clusters, which means that
# the inertia of the original centers used to generate the data is higher
# than the centers after KMeans, but it is logical since KMeans optimizes for
# it.
# 
# Reassignment ratio
# ==================
#
# We expect the fixed clusters to "get in the way" of clusters trying to move
# around to reach their cluster. Fortunately, the KMeans++ initialization
# usually takes care of this by positioning initialization centers in the best
# areas.


study_value = []
study_inert = []

for i in [0., 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    study_value.append(i)
    inert = evaluate(5000, 100, 20, 10, 10, 5, i, None)
    study_inert.append(inert)
    
plot('Reassignment ratio', 'Mean inertia over 100 runs',
     study_value, study_inert)

##############################################################################
# In the end, we observe that the reassignment ratio has absolutely no impact
# on the intertia which is reassuring.
# 
# Recenter every
# ==============
#
# The effect of this parameter has already been described above. In its
# highest value, this parameter leads to the following behavior: it initialize
# centers on the fixed center positions, let the KMeans run, and in the end
# move the centers closest to the fixed position on the fixed spots.

study_value = []
study_inert = []

for i in [None, 1, 2, 3, 5, 10, 20, 50, 100]:
    study_value.append(i if i is not None else 0)
    inert = evaluate(5000, 100, 20, 10, 10, 5, 0.1, i)
    study_inert.append(inert)
    
plot('Recenter every n iterations', 'Mean inertia over 100 runs', study_value, study_inert)


##############################################################################
# Surprisingly, this setting seems to give the best result. This result is a
# bit more surprising but a better inertia does not necessarily means a result
# closer to the data.
#
# Active learning like-experiment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# One core element of the experiment differs from active learning. In fact,
# in active learning, we do not expect to select a number of samples equal to
# the number of real cluster in the data. We also build our clustering over
# several iterations. We now try to do the same on generated data.

study_inert_kmeans = []
study_inert_ikmeans = []
study_inert_ikmeans_recenter = []

X, y, centers = make_blobs(n_samples=10000, centers=10, return_centers=True,
                           n_features=512, random_state=2)

fold = np.random.randint(0, 10, size=10000)

kmeans_centers = None
ikmeans_centers = None
ikmeans_recenter_centers = None

for i in range(10):
    kmeans = MiniBatchKMeans(n_clusters=10, random_state=i)
    ikmeans = IncrementalMiniBatchKMeans(n_clusters=10 * (i + 1), random_state=i)
    ikmeans_recenter = IncrementalMiniBatchKMeans(n_clusters=10 * (i + 1), random_state=i)

    ikmeans.fit(X[fold == i], fixed_cluster_centers=ikmeans_centers)
    ikmeans_centers = ikmeans.cluster_centers_

    ikmeans_recenter.fit(X[fold == i], fixed_cluster_centers=ikmeans_recenter_centers, recenter_every=100)
    ikmeans_recenter_centers = ikmeans_recenter.cluster_centers_

    kmeans.fit(X[fold == i])
    if kmeans_centers is None:
        kmeans_centers = kmeans.cluster_centers_
    else:
        kmeans_centers = np.vstack([kmeans_centers, kmeans.cluster_centers_])

    study_inert_kmeans.append(pairwise_distances_argmin_min(kmeans_centers, X)[1].mean())
    study_inert_ikmeans.append(pairwise_distances_argmin_min(ikmeans_centers, X)[1].mean())
    study_inert_ikmeans_recenter.append(pairwise_distances_argmin_min(ikmeans_recenter_centers, X)[1].mean())

plt.plot(range(10), study_inert_kmeans, label='KMeans inertia')
plt.plot(range(10), study_inert_ikmeans, label='Inceremental KMeans inertia')
plt.plot(range(10), study_inert_ikmeans_recenter, label='Inceremental KMeans R inertia')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Inertia')
plt.show()

##############################################################################
# To conclude, we observe that in an active-learning like settings, the best
# strategy in term of inertia consists in maintaining the clusters fixed.
