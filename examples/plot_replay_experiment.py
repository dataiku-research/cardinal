"""
Replay and experiment
=====================

Active Learning experiments can be long and costly. For this reason,
it is useful to be able to resume an experiment if an error happened.
To achieve that, cardinal allows to store intermediate variables,
such as selected samples, in a cache. Users can therefore resume
an interrupted experiment, using the ResumeCache, or replay an entire
experiment to perform additional computation such as metric using the
ReplayCache.
"""

import shutil
import os
import numpy as np
import dataset

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from cardinal.random import RandomSampler
from cardinal.uncertainty import MarginSampler
from cardinal.cache import ReplayCache
from cardinal.utils import GrowingIndex

##############################################################################
# Since we will be looking at the cache, we need a utility function to display
# a tree folder.

def print_folder_tree(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

#############################################################################
# We load the data and define the parameters of this experiment:  
#
# * ``batch_size`` is the number of samples that will be annotated and added to
#   the training set at each iteration,
# * ``n_iter`` is the number of iterations in our simulation

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
batch_size = 5
n_iter = 10

model = SVC(probability=True)

samplers = [
    ('margin', MarginSampler(model, batch_size))
]

#############################################################################
# We define our experiment in a dedicated function since we want to run it
# several times. We also create a dedicated exception that we will rise to
# simulate an interruption in the experiment.
#
# Note the use of the GrowingIndex utils that facilitate the handing of
# indices in an active learning experiment.


class ExampleError(Exception):
    pass


def run(force_failure=False, compute_metric=None):

    for sampler_name, sampler in samplers:

        config = dict(sampler=sampler_name)

        with ReplayCache('./cache', './cache.db', keys=config) as cache:

            index = GrowingIndex(X_train.shape[0])

            # Add at least one sample from each class
            index.add_to_selected([np.where(y_train == i)[0][0] for i in np.unique(y)])

            selected = cache.variable('selected', index.selected)

            for j, prev_selected in cache.iter(range(n_iter), selected.previous()):
                print('Computing iteration {}'.format(j))
                index.resume(prev_selected)

                model.fit(X_train[prev_selected], y_train[prev_selected])
                sampler.fit(X_train[prev_selected], y_train[prev_selected])
                index.add_to_selected(sampler.select_samples(X_train[index.non_selected]))
                selected.set(index.selected)

                if force_failure and j == 5:
                    raise ExampleError('Simulated Error')

            if compute_metric is not None:
                cache.compute_metric('metric', compute_metric, selected.previous(), selected.current())


#############################################################################
# We run this function and force an error to happen. We then see how the
# cache stores these values in a human readable way.
#
# We see that all selected indices have been kept up until the 4th iteration
# (since an error happened at iteration 5).

try:
    run(force_failure=True)
except ExampleError as e:
    print('ExempleError raised: ' + str(e))
print_folder_tree('./cache')

#############################################################################
# We run the same function without error. In this case, we see that the 4
# first iterations are skipped. The code is not even executed. Afterward,
# the cache contains the data for all iterations.

run()
print_folder_tree('./cache')

#############################################################################
# Being a bit paranoid, we would like to check what cardinal does. For that,
# we compute the batch size of each iteration. Fortunately, we have cached
# the variable `selected` and therefore, we can replay the experiment.


def calc_batch_size(previous, current):
    return current.sum() - previous.sum()


run(compute_metric=calc_batch_size)

for r in dataset.connect('sqlite:///cache.db')['metric'].all():
    if r['id'] == 1:
        print('\t'.join(r.keys()))
    print('\t'.join(map(str, r.values())))


#############################################################################
# We clean all the cache folder.

shutil.rmtree('./cache')
os.remove('./cache.db')