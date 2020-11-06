"""
Starting an Active Learning experiment
======================================

Each active learning experiment starts with zero labeled samples. In that
case, no supervised query strategy can be used. It is common to select the
first batch of samples at random however it is proposed in
`Diverse mini-batch Active Learning <https://arxiv.org/abs/1901.05954>`_
to use a clustering approach for the first batch.
This example shows how to perform this.
"""


##############################################################################
# We start with necessary imports and initializations

from matplotlib import pyplot as plt
import numpy as np
from time import time

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from cardinal.uncertainty import MarginSampler
from cardinal.clustering import KMeansSampler
from cardinal.random import RandomSampler
from cardinal.plotting import plot_confidence_interval
from cardinal.base import BaseQuerySampler
from cardinal.zhdanov2019 import TwoStepKMeansSampler
from cardinal.utils import pad_with_random

np.random.seed(7)

##############################################################################
# The parameters of this experiment are:
#
# * ``batch_size`` is the number of samples that will be annotated and added to
#   the training set at each iteration,
# * ``n_iter`` is the number of iterations in the simulation.
#
# We use the digits dataset and a RandomForestClassifier.

batch_size = 45
n_iter = 10

X, y = load_digits(return_X_y=True)
X /= 255.

model = RandomForestClassifier()

##############################################################################
# Core Active Learning Experiment
# -------------------------------
#
# We now compare our class sampler to Zhdanov, a simpler KMeans approach and, 
# of course, random. For each method, we measure the time spent at each iteration 
# and we plot the accuracy depending on the size of the labeled pool but also time spent.

starting_samplers = [
    ('KMeans', KMeansSampler(batch_size)),
    ('Random', RandomSampler(batch_size)),
]

samplers = [
    ('Zhdanov', TwoStepKMeansSampler(5, model, batch_size)),
    ('Margin', MarginSampler(model, batch_size)),
]

#figure_accuracies = plt.figure().number


for starting_sampler_name, starting_sampler in starting_samplers:
    for sampler_name, sampler in samplers:
    
        all_accuracies = []

        for k in range(10):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=500, random_state=k)

            accuracies = []

            # We use the starting sampler to kickstart the experiment
            selected = starting_sampler.fit(X_train).select_samples(X_train)

            # We use binary masks to simplify some operations
            mask = np.zeros(X_train.shape[0], dtype=bool)
            indices = np.arange(X_train.shape[0])
            mask[selected] = True

            # The classic active learning loop
            for j in range(n_iter):
                model.fit(X_train[mask], y_train[mask])

                # Record metrics
                accuracies.append(model.score(X_test, y_test))

                t0 = time()
                sampler.fit(X_train[mask], y_train[mask])
                selected = sampler.select_samples(X_train[~mask])
                mask[indices[~mask][selected]] = True

            all_accuracies.append(accuracies)
    
        x_data = np.arange(10, batch_size * (n_iter - 1) + 11, batch_size)
        plot_confidence_interval(x_data, all_accuracies, label='{} + {}'.format(starting_sampler_name, sampler_name))


plt.xlabel('Labeled samples')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()

plt.show()

##############################################################################
# Discussion
# ----------

