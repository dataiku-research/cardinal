import numpy as np
from cardinal.utils import ActiveLearningSplitter


def test_active_learning_splitter():

    splitter = ActiveLearningSplitter(100)
    y = np.zeros(100)
    y[1] = 1
    y[2] = 2

    splitter.initialize_with_random(4, at_least_one_of_each_class=y)
    assert(np.in1d([1, 2], np.where(splitter.selected)).all())

    assert(splitter.current_iter == 0)
    splitter.add_batch([3, 5, 7])
    assert(splitter.current_iter == 1)
    
