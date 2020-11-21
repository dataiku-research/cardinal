"""
Running and resuming an experiment
==================================

Active Learning experiments can be long and costly. For this reason,
it is useful to be able to resume an experiment if an error happened.
To achieve that, cardinal allows to store intermediate variables,
such as selected samples, in a cache called ResumeCache. Let us see
how to use it in this example.
"""
import os
import shutil

import numpy as np
import dataset

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from cardinal.uncertainty import MarginSampler
from cardinal.cache import ResumeCache, ShelveStore
from cardinal.utils import SampleSelector

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

CACHE_PATH = './cache'
DATABASE_PATH = './cache.db'

value_store = ShelveStore(DATABASE_PATH)

#############################################################################
# We define our experiment in a dedicated function since we want to run it
# several times. We also create a dedicated exception that we will rise to
# simulate an interruption in the experiment.
#
# Note the use of the GrowingIndex utils that facilitate the handing of
# indices in an active learning experiment.


class ExampleError(Exception):
    pass


def run(force_failure=False):

    for sampler_name, sampler in samplers:

        experiment_config = dict(sampler=sampler_name)

        with ResumeCache(CACHE_PATH, value_store, keys=experiment_config) as cache:

            # Create a selector with one sample from each class and persist it
            init_selector = SampleSelector(X_train.shape[0])
            init_selector.add_to_selected([np.where(y_train == i)[0][0] for i in np.unique(y)])
            selector = cache.persisted_value('selector', init_selector)

            for j, prev_selector in cache.iter(range(n_iter), selector.previous()):
                print('Computing iteration {}'.format(j))

                model.fit(X_train[prev_selector.selected], y_train[prev_selector.selected])
                sampler.fit(X_train[prev_selector.selected], y_train[prev_selector.selected])
                prev_selector.add_to_selected(sampler.select_samples(X_train[prev_selector.non_selected]))

                selector.set(prev_selector)

                cache.log_value('accuracy', model.score(X_test, y_test))

                if force_failure and j == 5:
                    raise ExampleError('Simulated Error')


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
# During the experiment, we have cached the accuracy value. Let us see how
# to get it and plot it.

from matplotlib import pyplot as plt


accuracy = value_store.get('accuracy')

plt.plot(accuracy['iteration'], accuracy['value'])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Evolution of accuracy during active learning experiment on Iris dataset')
plt.show()


#############################################################################
# We clean all the cache folder.

#hutil.rmtree(CACHE_PATH)
#os.remove(DATABASE_PATH)