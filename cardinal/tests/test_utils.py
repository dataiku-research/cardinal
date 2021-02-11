from cardinal.utils import ActiveLearningSplitter
import numpy as np


def test_active_learning_splitter():

    X = np.random.random((15, 2))

    splitter = ActiveLearningSplitter(X, test_size=4)
    init, = splitter.get_batch()
    assert(init.shape == (0, 2))

    splitter.add_batch([0, 1, 3])
    iter_0, = splitter.get_batch()
    assert(iter_0.shape == (3, 2))

    splitter.add_batch([2, 3])
    iter_1, = splitter.get_batch()
    assert(iter_1.shape == (2, 2))

    train, = splitter.get_train()
    assert(train.shape == (11, 2))

    test, = splitter.get_test()
    assert(test.shape == (4, 2))

    selected, = splitter.get_selected()
    assert(selected.shape == (5, 2))

    non_selected, = splitter.get_non_selected()
    assert(non_selected.shape == (6, 2))
