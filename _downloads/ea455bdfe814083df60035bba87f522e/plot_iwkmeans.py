
"""
Noisy boundary 2
================

Can a noisy boundary tamper with an active learning process? For this test,
we select a very simple classification problem. However, this problem is made
more difficult by the existence of a noisy boundary where samples are
indistinguishable from each other. Let us see if this samplers fall for this
"honey pot".
"""

from copy import deepcopy

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from cardinal.utils import ActiveLearningSplitter
from cardinal.uncertainty import ConfidenceSampler
from cardinal.random import RandomSampler
from cardinal.clustering import KMeansSampler
from cardinal.plotting import smooth_lines


n_samples = 10000
n_features = 100
n_classes = 10
n_blobs = 100
n_noisy_blobs = 10
seed = 2
batch_size = 20

if n_noisy_blobs % 2 != 0:
    raise ValueError("Noisy blob number must be even")

if n_blobs % n_classes != 0:
    raise ValueError('Please specify a uniform amount of blobs per class')

ppb = n_samples // n_blobs
spb = [ppb * 2] * (n_noisy_blobs // 2) + [ppb] * (n_blobs - n_noisy_blobs)


X, y_ = make_blobs(spb, random_state=seed)

# Now attribute labels to noisy blobs
for i in range(n_blobs - n_noisy_blobs, n_blobs - n_noisy_blobs // 2):
    # Chose half of the points of the noisy blob to assign them a new label
    idx = np.where(y_ == i)[0]
    np.random.shuffle(idx)
    y_[idx[:idx.shape[0] // 2]] = i + (n_noisy_blobs // 2)

assert(np.unique(y_).shape[0] == n_blobs)

# Now assign blobs to classes
classes = np.arange(n_classes).repeat(n_blobs // n_classes)
np.random.shuffle(classes)
y = classes[y_]




clf = RandomForestClassifier()
init_spl = ActiveLearningSplitter.train_test_split(X.shape[0], test_size=0.2, shuffle=True, stratify=y, random_state=seed)

init_random_spl = deepcopy(init_spl)
np.random.seed(seed)
init_idx = np.hstack([
    np.random.choice(np.where(y[init_spl.train] == 0)[0], size=batch_size),
    np.random.choice(np.where(y[init_spl.train] == 1)[0], size=batch_size),
])
init_random_spl.initialize_with_indices(init_idx)

##############################################################################
# This function runs the experiment. It is a class active learning setting.

def evaluate(acc, cnt, name, sampler, init_spl, n_iter=20):
    spl = deepcopy(init_spl)
    g_acc = []
    n_cnt = []
    
    for _ in range(n_iter):
        clf.fit(X[spl.selected], y[spl.selected])
        sampler.fit(X[spl.selected], y[spl.selected])
        spl.add_batch(sampler.select_samples(X[spl.non_selected]))
        g_acc.append(accuracy_score(y[spl.test], clf.predict(X[spl.test])))
        n_cnt.append((y_[spl.test] >= n_blobs - n_noisy_blobs).mean())
    
    acc[name] = g_acc
    cnt[name] = n_cnt

##############################################################################
# We create a figure to track both the global accuracy with random and noisy
# initialization and display the results for 3 very common samplers.

acc = dict()
cnt = dict()

evaluate(acc, cnt, 'Confidence Sampler', ConfidenceSampler(clf, batch_size=batch_size, assume_fitted=True), init_random_spl)
evaluate(acc, cnt, 'Random Sampler', RandomSampler(batch_size=batch_size, random_state=0), init_random_spl)
evaluate(acc, cnt, 'KMeans Sampler', KMeansSampler(batch_size=batch_size), init_random_spl)


plt.figure()
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
for name in acc:
    data = acc[name]
    plt.plot(np.arange(len(data)), data, label=name)

gr_ax = plt.gca()
gr_ax.legend()
#smooth_lines(axis=gr_ax, k=2)

plt.figure()
plt.ylabel('Ratio of noisy samples')
plt.xlabel('Iteration')
for name in cnt:
    data = cnt[name]
    plt.plot(np.arange(len(data)), data, label=name)

gr_ax = plt.gca()
gr_ax.legend()
#smooth_lines(axis=gr_ax, k=2)

plt.show()
