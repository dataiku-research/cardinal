"""
Lowest confidence vs. KMeans sampling
=====================================

In this example, we show the usefulness of diversity-based approaches using a
toy example where a very unlucky initialization makes lowest confidence
approach underperform.

"""


##############################################################################
# Those are the necessary imports and initialiaztion

from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC

from cardinAL.uncertainty import ConfidenceSampler
from cardinAL.clustering import KMeansSampler

np.random.seed(7)

##############################################################################
# Parameters of our experiment:
# * _n_ is the number of points in the sumulated data
# * _batch_size_ is the number of samples that will be annotated and added to
#   the training set at each iteration
# * _n_iter_ is the number of iterations in our simulation
#
# We simulate data where the samples of one of the class are scattered in 3
# blobs, one of them being far away from the two others. We also select an
# initialization index where no sample from the far-away sample is initially
# selected. This will force the decision boundary to stay far from that cluster
# and thus "trick" the lowest confidence method.

n = 28
batch_size = 4
n_iter = 5

X, y = make_blobs(n_samples=n, centers=[(1, 0), (0, 1), (2, 2), (4, 0)],
                  random_state=0, cluster_std=0.2)
    
# We select samples in clusters 0, 1 and 2. Cluster 3 will be ignored by uncertainty sampling
init_idx = [i for j in range(3) for i in np.where(y == j)[0][:2]]
y[y > 1] = 1

model = SVC(kernel='linear', C=1E10, probability=True)

##############################################################################
# This helper function plots our simulated points in red and blue. The one that
# are not in the training set are faded. We also plot the linear separation
# estimated by the SVM.

def plot(a, b, score, selected):
    plt.xlabel('Accuracy {}%'.format(int(score * 100)), fontsize=10)

    # Plot not selected first in low alpha, then selected
    for l, s in [(0, False), (1, False), (0, True), (1, True)]:
        alpha = 1. if s else 0.3
        color = 'tomato' if l == 0 else 'royalblue'
        mask = np.logical_and(selected == s, l == y)
        plt.scatter(X[mask, 0], X[mask, 1], c=color, alpha=alpha)
        
    # Plot the separation margin of the SVM
    x_bounds = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
    plt.plot(x_bounds, a * x_bounds + b)


##############################################################################
# Core active learning experiment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As presented in the introduction, this loop represents the active learning
# experiment. At each iteration, the model is learned on all labeled data to
# measure its performance. Then, the model is inspected to find out the samples
# on which its confidence is low. This is done through cardinAL samplers.

samplers = [
    ('Lowest confidence', ConfidenceSampler(model, batch_size)),
    ('KMeans', KMeansSampler(batch_size))
]

plt.figure(figsize=(10, 4))

for i, (sampler_name, sampler) in enumerate(samplers):
    mask = np.zeros(n, dtype=bool)
    indices = np.arange(n)
    mask[init_idx] = True

    for j in range(n_iter):
        model.fit(X[mask], y[mask])
        sampler.fit(X[mask], y[mask])
        w = model.coef_[0]
        
        plt.subplot(len(samplers), n_iter, i * n_iter + j + 1)
        plot(-w[0] / w[1], - model.intercept_[0] / w[1], model.score(X, y), mask.copy())

        selected = sampler.select_samples(X[~mask])
        mask[indices[~mask][selected]] = True

        if j == 0:
            plt.ylabel(sampler_name)
        plt.axis('tight')
        plt.gca().set_xticks(())
        plt.gca().set_yticks(())
        if i == 0:
            plt.gca().set_title('Iteration {}'.format(j), fontsize=10)

plt.tight_layout()
plt.subplots_adjust(top=0.86)
plt.gcf().suptitle('Classification accuracy of random and uncertainty active learning on simulated data', fontsize=12)
plt.show()
