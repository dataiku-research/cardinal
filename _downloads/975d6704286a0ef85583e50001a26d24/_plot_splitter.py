"""
Active Learning Splitter
========================

Active learning experiments require heavy manipulation of indices
for train vs test, train being divided in labeled and unlabeled,
labeled being divided again in train vs validation... All those
index manipulation is tedious, error-prone, and can lead to repeated
unwanted copies of data.

Cardinal provides a splitter easing all indices manipulation. It
also comes with handy procedures that make active learning
experiments easier.

Train vs test split
-------------------

The first in an active learning experiment is to split the data
in train and test. The training split is used to simulate the
pool of unlabeled data while the testing split is a left-out
set of samples used to evaluate the sampler. Cardinal offers 
two ways of creating this split in the splitter:
* `ActiveLearningSplitter(n_samples, test_index=[...])` allows
  to create a splitter with no test, or by directly specifying
  the indices.
* `ActiveLearningSplitter.train_test_split` maps directly
  scikit-learn's function to create the train/test split inside
  the splitter.

See it in action below!
"""

from cardinal.utils import ActiveLearningSplitter
import numpy as np


no_test_splitter = ActiveLearningSplitter(100)
print('Creating a splitter without test set')
print('Training set size: ', no_test_splitter.train.sum())
print('Testing set size: ', no_test_splitter.test.sum())
print()

with_test_splitter = ActiveLearningSplitter(100, test_index=[10, 20, 30])
print('Creating a splitter with 3 test samples')
print('Training set size: ', with_test_splitter.train.sum())
print('Testing set size: ', with_test_splitter.test.sum())
print('Test indices: ', np.where(with_test_splitter.test)[0])
print()

sklearn_test_splitter = ActiveLearningSplitter.train_test_split(100, test_size=20)
print('Creating a splitter with sklearn\'s train_test_split')
print('Training set size: ', sklearn_test_splitter.train.sum())
print('Testing set size: ', sklearn_test_splitter.test.sum())
print()


##############################################################################
# Sampling the first batch
# ------------------------
#
# The first batch of an active learning experiment is always special since
# it is not selected through the active learning procedure itself. Most works
# use a random selection of samples for this first batch. Some work used
# an initilization through KMeans. Cardinal allows to use both of those. On
# top of this, cardinal allows to ensure that at least one sample per class
# is selected in the random selection. This is made to prevent label addition
# during the experiment that can be cumbersome to handle.