# Muscovy duck is known through the meme of "Judgemental bird"


import itertools
import os
import numpy as np
import shutil
from sklearn.utils import check_random_state


class IncompatibleStepError(Exception):
    pass

def robust_load(path):
    if os.path.isdir(path):
        # This is a lazy list
        data = LazyList(path)
        return data

    # Numpy sometimes does weird things. This contains workarounds
    data = np.load(path, allow_pickle=True)
    try:
        if data.dtype.char == 'O' and len(data.shape) == 0:
            data = data.min()
    except:
        pass
    return data


class LazyList():

    def __init__(self, filepath):
        self.filepath = filepath
        self._list = None
        self._loaded = False

    def _check_is_loaded(self):
        if self._loaded:
            return

        self._list = []
        if os.path.exists(self.filepath):
            # Each file in the folder is an element of the list
            i = 0
            path = os.path.join(self.filepath, 'el_{}'.format(i))
            while os.path.exists(path):
                self._list.append(path)
                i += 1
                path = os.path.join(self.filepath, 'el_{}'.format(i))
        else:
            os.makedirs(self.filepath)
        self._loaded = True

    def __len__(self):
        self._check_is_loaded()
        return len(self._list)

    def save_append(self, data):
        self._check_is_loaded()
        next_item = len(self)
        path = os.path.join(self.filepath, 'el_{}'.format(next_item))
        np.save(path, data)
        self._list.append(path)

    def __getitem__(self, index):
        self._check_is_loaded()
        path = self._list[index]
        return robust_load(path)


class Bencher:

    def __init__(self, steps, cache_dir, lazy_loading_prefix='ll_'):
        """Initialize the bencher with the experiment

        Steps is a list of (string, boolean). The boolean indicates if the
        result of the step must be persisted on disk.
        """
        self.steps = [step for step, _ in steps]
        self.steps_func = dict()
        self.should_persist = dict(steps)
        for step, _ in steps:
            self.steps_func[step] = dict()
        
        self.cache = dict()
        self.cache_dir = cache_dir
        self.reducers = dict()

    def register_step(self, step_id, step_name, func):
        """Register a new possible step for the benchmark
        """
        if not step_id in self.steps_func:
            raise ValueError('Step {} is unknown. Please chose a step among {}'.format(step_id, ' '.join(list(self.steps_func.keys()))))
        self.steps_func[step_id][step_name] = func

    def register_reducer(self, reducer_name, func):
        self.reducers['reducer_name'] = func

    def get_lazy_list(self, var_id):
        return LazyList(os.path.join(self._cache_key, var_id))
    
    def run(self):
        all_steps_name = [list(self.steps_func[step].keys()) for step in self.steps]
        
        final_results = dict()

        for steps_name_list in itertools.product(*all_steps_name):
            # We are going to chain the application of each step

            data = {}
            steps_so_far = []
            new_data = None

            for step_id, step_name in zip(self.steps, steps_name_list):
                print(step_id, step_name)
                steps_so_far.append(step_name)

                cache_key = '_'.join(steps_so_far)
                self._cache_key = cache_key

                # 1st option, the result is already in memory
                if step_id in self.cache:
                    cache_name, cache_data = self.cache[step_id]
                    if cache_name != cache_key:
                        # It is a result cached from another computation, discard it
                        del self.cache[step_id]
                    else:
                        new_data = cache_data
                        data.update(new_data)
                        continue
                
                # 2nd option, the result is cached on the disk
                if self.should_persist[step_id]:
                    cache_path = os.path.join(self.cache_dir, cache_key)
                    if os.path.exists(cache_path):
                        new_data = dict()
                        for d, _, fs in os.walk(cache_path):
                            for f in fs:
                                new_data[f.split('.')[0]] = robust_load(os.path.join(d, f))
                        data.update(new_data)
                        continue

                # 3rd option, we have to compute the result
                new_data = self.steps_func[step_id][step_name](data)

                if self.should_persist[step_id]:
                    cache_path = os.path.join(self.cache_dir, cache_key)
                    os.makedirs(cache_path)
                    try:
                        for key in new_data:
                            data = new_data[key]
                            if isinstance(data, LazyList):
                                # LazyList saves on the spot
                                pass
                            np.save(os.path.join(cache_path, key + '.npy'), new_data[key])
                    except:
                        print('Error saving cache. Please run this command to clean and relaunch:')
                        print('rm -rf {}'.format(cache_path))
                
                self.cache[step_id] = (step_name, new_data)
                data.update(new_data)
            
            final_results[steps_name_list] = new_data

        for _, func in self.reducers.items():
            func(final_results)


class SeedingStep:

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, data):
        random_state = check_random_state(int(self.seed))
        return dict(random_state=random_state)


class ValueStep:

    def __init__(self, value):
        self.value = value

    def __call__(self, data):
        return self.value



# Get dataset
def get_sklearn_mnist(data):

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler


    # Load data from https://www.openml.org/d/554
    # 70000 samples
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    return dict(X=X, y=y)


def get_mxnet_mnist(data):

    import mxnet as mx
    from mxnet import gluon, autograd, ndarray

    train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.MNIST(train=True, transform=lambda data, label: (data.astype(np.float32)/255, label)),
        batch_size=np.inf, shuffle=False)

    X_train, y_train = list(train_data)[0]
    X_train = X_train.reshape(X_train.shape[0], -1)

    permutation = data['random_state'].permutation(X_train.shape[0])

    X_train = X_train[permutation].asnumpy()
    y_train = y_train[permutation].asnumpy()

    test_data = mx.gluon.data.DataLoader(
        mx.gluon.data.vision.MNIST(train=False, transform=lambda data, label: (data.astype(np.float32)/255, label)),
        batch_size=np.inf, shuffle=False)

    X_test, y_test = list(test_data)[0]
    X_test = X_test.reshape(X_test.shape[0], -1).asnumpy()
    y_test = y_test.asnumpy()

    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def get_keras_mnist(data):

    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return dict(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)


def get_keras_cifar(data):

    from keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    return dict(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)


def random_sampler_step(data):
    from cardinAL.random import RandomSampler

    return dict(sampler=RandomSampler(batch_size=data['batch_size'], random_state=data['random_state']))

def sampler_step(clazz, *args, **kwargs):
    return lambda data: dict(sampler=clazz(batch_size=data['batch_size'], *args, **kwargs))
