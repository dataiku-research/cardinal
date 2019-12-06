"""
Hello
=====

This is a test
"""


from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import numpy as np
from cardinAL.random import RandomSampler
from cardinAL.uncertainty import ConfidenceSampler
from copy import copy
import matplotlib.animation as animation
from matplotlib.cm import tab20


# Variables
n = 30
batch_size = 2
n_iter = 6

# Initialize the data
np.random.seed(8)

X, y = make_blobs(n_samples=n, centers=2,
                  random_state=0, cluster_std=0.80)

model = SVC(kernel='linear', C=1E10, probability=True)

# Vars for plotting purpose
min_X = np.min(X[:, 0])
max_X = np.max(X[:, 0])

# We need to run the simulation in advance
samplers = [
    ('Random', RandomSampler(batch_size=batch_size, random_state=0)),
    ('Lowest confidence', ConfidenceSampler(model, batch_size))
]

def plot(a, b, score, selected):
    plt.xlabel('Accuracy {}%'.format(int(score * 100)), fontsize=10)

    y_ = y.copy()
    y_[~selected] = 3 - y_[~selected]

    plt.scatter(X[:, 0], X[:, 1], c=y_, cmap=tab20)

    f = lambda x: a * x + b
    xx = np.asarray([min_X, max_X])
    plt.plot(xx, f(xx))


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
