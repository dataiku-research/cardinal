"""
Lowest confidence vs. Random sampling
=====

The simplest way ton convince ourselves that active learning actually
works is to first test it on simulated data. In this example, we will
generate a simple classification task and see how active learning allows
ont to converge faster toward the best result.
"""


##############################################################################
# Those are the necessary imports and initialiaztion

from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC

from cardinAL.random import RandomSampler
from cardinAL.uncertainty import ConfidenceSampler


np.random.seed(8)

##############################################################################
# Parameters of our experiment:
# * _n_ is the number of points in the sumulated data
# * _batch_size_ is the number of samples that will be annotated and added to
#   the training set at each iteration
# * _n_iter_ is the number of iterations in our simulation
#
# Our simulated data is composed of 2 clusters that are very close to each other
# but linearly separable. We use as simple SVM classifier as it is a basic classifer.


n = 30
batch_size = 2
n_iter = 6

X, y = make_blobs(n_samples=n, centers=2,
                  random_state=0, cluster_std=0.80)

model = SVC(kernel='linear', C=1E10, probability=True)

##############################################################################
# This helper function plots our simulated points in red and blue. The one that
# are not in the training set are faded. We also plot the linear separation
# estimated by the SVM.

def plot(a, b, score, selected):
    plt.xlabel('Accuracy {}%'.format(int(score * 100)), fontsize=10)

    # We map our prediction to 4 values:
    # 0 is dark blue and designates samples of class 1 in the training set
    # 1 is dark red and designates samples of class 2 in the training set
    # 2 is light red and designates samples of class 2 *not* in the training set
    # 3 is light blue and designates samples of class 1 *not* in the training set
    y_ = y.copy()
    y_[~selected] = 3 - y_[~selected]

    plt.scatter(X[:, 0], X[:, 1], c=y_, cmap='tab20')

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
    ('Random', RandomSampler(batch_size=batch_size, random_state=0)),
    ('Lowest confidence', ConfidenceSampler(model, batch_size))
]

plt.figure(figsize=(10, 4))

for i, (sampler_name, sampler) in enumerate(samplers):
    # We force having one sample in each class for the init
    init_idx = [np.where(y == 0)[0][0], np.where(y == 1)[0][0]]

    selected = np.zeros(n, dtype=bool)
    selected[init_idx] = True

    for j in range(n_iter):
        model.fit(X[selected], y[selected])
        sampler.fit(X[selected], y[selected])
        w = model.coef_[0]
        
        plt.subplot(len(samplers), n_iter, i * n_iter + j + 1)
        plot(-w[0] / w[1], - model.intercept_[0] / w[1], model.score(X, y), selected.copy())

        new_selected = sampler.predict(X[~selected])
        selected[~selected] = new_selected

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