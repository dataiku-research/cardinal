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

from cardinAL.uncertainty import UncertaintySampler, MarginSampler, EntropySampler


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


n = 250
batch_size = 10
n_iter = 10

# model = SVC(kernel='linear', C=1E10, probability=True)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_features="auto")


##############################################################################
# Core active learning experiment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As presented in the introduction, this loop represents the active learning
# experiment. At each iteration, the model is learned on all labeled data to
# measure its performance. Then, the model is inspected to find out the samples
# on which its confidence is low. This is done through cardinAL samplers.

samplers = [
    ('Lowest confidence', UncertaintySampler(model, batch_size)),
    ('Smaller margin', MarginSampler(model, batch_size)),
    ('Highest entropy', EntropySampler(model, batch_size))
]


for centers in [2, 3, 5, 8, 12, 20]:

    X, y = make_blobs(n_samples=n, centers=centers,
                      random_state=0, cluster_std=0.80)

    # We take 10 samples in each class as init
    init_idx = [x for i in range(centers) for x in np.where(y == i)[0][:1]]

    res = dict()
    plt.figure()

    for i, (sampler_name, sampler) in enumerate(samplers):
        print(sampler_name)

        selected = np.zeros(n, dtype=bool)
        selected[init_idx] = True

        res[sampler_name] = []
        
        for j in range(n_iter):
            model.fit(X[selected], y[selected])
            sampler.fit(X[selected], y[selected])
            new_selected = sampler.predict(X[~selected])
            selected[~selected] = new_selected

            res[sampler_name].append(model.score(X, y))

            #if j == 0:
            #    plt.ylabel(sampler_name)
            #plt.axis('tight')
            #plt.gca().set_xticks(())
            #plt.gca().set_yticks(())
            #if i == 0:
            #    plt.gca().set_title('Iteration {}'.format(j), fontsize=10)
        plt.plot(res[sampler_name], label=sampler_name)
    print(centers, res)
plt.show()

#plt.tight_layout()
#plt.subplots_adjust(top=0.86)
#plt.gcf().suptitle('Classification accuracy of random and uncertainty active learning on simulated data', fontsize=12)
#plt.show()