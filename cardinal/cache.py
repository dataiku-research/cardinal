from importlib import import_module
import dataset
import shelve
import copy
import numpy as np
import pickle
import os
import shutil
import itertools
from pathlib import Path
import json
import time
import pandas as pd
from abc import ABC, abstractmethod


class HashableDict(dict):
    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


class ValueStore(ABC):

    @abstractmethod
    def __init__(self, filename):
        pass

    @abstractmethod
    def _store(self, name, value, keys):
        pass

    @abstractmethod
    def _sync(self):
        pass

    @abstractmethod
    def get(self, name):
        pass

    @abstractmethod
    def close(self):
        pass


class ShelveStore(ValueStore):

    def __init__(self, filename, writeback=False):
        self.writeback = writeback
        import shelve
        if not filename.endswith('.db'):
            raise ValueError('File extension must be .db with shelve backend')
        filename = filename[:-3]  # Shelve adds a .db extension automatically
        self._conn = shelve.open(filename, writeback=writeback)

    def _store(self, name, value, **keys):
        keys = HashableDict(keys)
        if self.writeback:
            if not name in self._conn:
                self._conn[name] = dict()
            self._conn[name][keys] = value
        else: 
            d = dict()
            if name in self._conn:
                d = self._conn[name]
            d[keys] = value
            self._conn[name] = d

    def get(self, name):
        self._conn.sync()
        if name not in self._conn:
            return pd.DataFrame([])
        return pd.DataFrame.from_records([dict(value=v, **k) for k, v in self._conn[name].items()])

    def _sync(self):
        self._conn.sync()

    def close(self):
        self._conn.close()



class SqliteStore(ValueStore):

    def __init__(self, filename):
        import dataset
        self._conn = dataset.connect('sqlite:///' + filename)

    def _store(self, name, value, **keys):
        table = self._conn[name]
        table.upsert(dict(value=value, **keys), list(keys.keys()))

    def get(self, name):
        if name not in self._conn:
            return pd.DataFrame([])
        return pd.DataFrame(list(self._conn[name].all())).drop('id', axis=1)

    def _sync(self):
        self._conn.sync()

    def close(self):
        self._conn.close()


class ResumeCache:

    _clear_outdated_variables = True

    def __init__(self, cache_dir, value_store, keys={}):
        self.cache_dir = Path(cache_dir).joinpath(*[Path(k) / Path(str(v)) for k, v in keys.items()])
        self.value_store = value_store
        self.keys = keys
        self._current_iter = -1

    def __enter__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, type, value, traceback):
        ReplayCache.cache = None

    def persisted_value(self, name, init_value):
        """Value persisted in cache for resuming or replaying

        A persisted value is kept in memory and on disk for a short duration to allow
        for resuming (using ResumeCache), or forever to allow for replaying the
        experiment (ReplayCache).

        Args:
            name: The name of the persisted variable.
            init_value: The value attributed to the variable.

        Returns:
            A variable object allowing to access persisted data.
        """
        return Variable(name, init_value, cache=self.cache_dir, clear_outdated=self._clear_outdated_variables)

    def log_value(self, key, value, iteration='auto', **kwargs):
        if type(value).__module__ == np.__name__:
            value = value.item()
        log_keys = self.keys.copy()
        if iteration == 'auto' and self._current_iter >= 0:
            log_keys['iteration'] = self._current_iter
        elif iteration is not None:
            log_keys['iteration'] = iteration
        log_keys.update(kwargs)
        self.value_store._store(key, value, **log_keys)

    def iter(self, iterator, *variables):

        if self._current_iter == -2:
            raise ValueError('This experiment has already run')

        json_path = self.cache_dir / "progress.json"

        if json_path.exists():
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
                self._current_iter = max([int(k) for k in data.keys()])

        for i, item in enumerate(iterator):
            if i <= self._current_iter:
                continue
            self._current_iter = i
            t0 = time.time()

            yield tuple([item] + [variable._get(i + offset) for variable, offset in variables])

            duration = time.time() - t0

            for variable, _ in variables:
                variable._persist(i)

            data = {}
            if json_path.exists():
                with open(json_path, "r") as json_file:
                    data = json.load(json_file)

            data[i] = duration

            with open(json_path, "w") as json_file:
                json.dump(data, json_file)

            for variable, _ in variables:
                variable._clear(i)
            
            self.value_store._sync()
        
        self._current_iter = -2


class ReplayCache(ResumeCache):

    _clear_outdated_variables = False

    def persisted_value(self, name, init_value):
        return Variable(name, init_value, cache=self.cache_dir, clear_outdated=False)

    def _is_variable(self, v):
        return (isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], Variable))

    def compute_metric(self, name, func, *args, **kwargs):
        if not self._current_iter == -2:
            raise ValueError('Please run your experiment before computing metrics')

        json_path = self.cache_dir / "progress.json"
        with open(json_path, "r") as json_file:
            progress = json.load(json_file)
        max_iter = max([int(s) for s in progress])

        var_args = [i for i, a in enumerate(args) if self._is_variable(a)]
        var_kwargs = [k for k in kwargs if self._is_variable(kwargs[k])]

        for i in range(max_iter + 1):
            cargs = list(args)
            for j in var_args:
                cargs[j] = args[j][0]._get(args[j][1] + i)
                args[j][0]._clear(i)
            ckwargs = kwargs.copy()
            for k in var_kwargs:
                ckwargs[k] = kwargs[k][0]._get(kwargs[k][1] + i)
                ckwargs[k]._clear(i)
            v = func(*cargs, **ckwargs)
            self.log_value(name, v, iteration=i)


class Variable():

    def __init__(self, name, init_value, cache='./cache', clear_outdated=True):
        self.name = name
        self.init_value = init_value
        self.cache = Path(cache)
        self.clear_outdated = clear_outdated
        self._lifetime = 1
        self._values = dict()
    
    def previous(self, offset=-1):
        self._lifetime = max(-offset, self._lifetime)
        return self, offset

    def current(self):
        return self, 0

    def set(self, value):
        self._cached = value

    def _exists(self, iteration):
        return iteration < 0 or iteration in self._values or (self.cache / str(iteration) / self.name).exists()

    def _load(self, iteration):
        path = (self.cache / str(iteration) / self.name)

        # Check if numpy
        try:
            self._values[iteration] = np.load(path)
            return
        except ValueError:
            pass

        with open(path, 'rb') as filedesc:
            self._values[iteration] = pickle.load(filedesc)

    def _get(self, iteration):
        if iteration < 0:
            return self.init_value

        if not iteration in self._values:
            # Load from disk
            self._load(iteration)
        
        return self._values[iteration]

    def _persist(self, iteration):

        self._values[iteration] = self._cached
        
        path = (self.cache / str(iteration))
        path.mkdir(parents=True, exist_ok=True)
        path = path / self.name
        with open(path, 'wb') as f:
            if type(self._cached).__module__ == np.__name__:
                np.save(f, self._cached)
            else:
                pickle.dump(self._cached, f)

    def _clear(self, iteration):

        # Delete unnecessary objects
        to_delete = []
        for key in self._values:
            if key <= iteration - self._lifetime:
                to_delete.append(key)
        for key in to_delete:
            del self._values[key]
            if self.clear_outdated:
                path = (self.cache / str(key)) / self.name
                path.unlink()
