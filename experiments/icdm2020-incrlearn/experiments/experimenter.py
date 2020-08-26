import sys
import json
from importlib import import_module
import dataset
import hashlib
import copy
import numpy as np
import pickle
import os
import shutil
import itertools


# Description of the framework
#
# Context
# -------
# The base concept of this framework is the notion of context. A context
# is created in a given situation (experiment about xxx, iteration n).
# When created, a context inherits the data of the previous context. When
# a context is destroyed, two things can happen:
# - this is an old context, its data is no more needed for further
#   computation. All non-persisted variables are destroyed to save space
# - this is a context which data may be used for the next iteration. All
#   its data is kept to be able to resume computation in case of failure
#
# Experiment
# ----------
# An experiment is the overarching manager of contexts. It proposes a
# convenient for-loop to switch between contexts and takes care of caching
# or deleting old contexts.
#
# Variable
# --------
# A variable is a proxy on a value. It stores no value but can access the
# current ones by doing var -> experiment -> current context -> value.
# Through the Variable.val property, the user accesses the previous value
# of the variable.
# If persisted, the data of the variable can be accessed after the experiment
# If not, it is destroyed when it is not needed anymore


def _paired_iter(gen_or_list):
    nil = object()
    prev = nil
    for item in gen_or_list:
        if not prev is nil:
            yield (prev, item)
        prev = item


class Variable():

    def __init__(self, experiment, key):
        self.experiment = experiment
        self.key = key

    @property
    def val(self):
        return self.experiment._get_variable(self.key)

    @val.setter
    def val(self, value):
        self.experiment._set_variable(self.key, value)


class LazyValue():

    unset = object()

    def __init__(self, scope, name, value=unset):
        self.scope = scope
        self.name = name
        self.value = value

    def _load_value(self):
        for subdir in ['vars', 'tmp']:
            filebase = os.path.join(self.scope.scope_folder, subdir, self.name)
            if os.path.exists(filebase + '.npy'):
                self.value = np.load(filebase + '.npy')
                return
            elif os.path.exists(filebase + '.pkl'):
                with open(filebase + '.pkl', 'rb') as filedesc:
                    self.value = pickle.load(filedesc)
                    return
            else:
                raise ValueError('Could not load variable')
    
    def get_value(self):
        if self.value is self.unset:
            self._load_value()
        return self.value

    def persist(self):
        if self.value is self.unset:
            raise ValueError('Unset value cannot be persisted')


class Scope():

    def __init__(self, cache_folder, seed, id):
        self.cache_folder = cache_folder
        self.id = id
        self.modified_variables = set()
        self.seed = seed
        self.scope_folder = os.path.join(cache_folder, str(seed), *id)
        if not os.path.exists(self.scope_folder):
            os.makedirs(self.scope_folder)

    def tag_modified_variable(self, var_id):
        self.modified_variables.add(var_id)

    def is_completed(self):
        if not os.path.exists(os.path.join(self.scope_folder, '..', 'completed')):
            return False
        with open(os.path.join(self.scope_folder, '..', 'completed'), 'r') as completed:
            for completed_step_name in completed:
                if self.id[-1] == completed_step_name.strip():
                    return True
        return False

    def _load_data_from_scope(self, variables, scope):
        var_folder = os.path.join(scope.scope_folder, 'vars')
        if not os.path.exists(var_folder):
            return
        for filename in os.listdir(var_folder):
            filepath = os.path.join(var_folder, filename)
            if not os.path.isfile(filepath):
                continue
            name = filename.split('.')[0]
            variables[name] = LazyValue(scope, name)

    def load_data(self, variables):
        self._load_data_from_scope(variables, self)

    def load_previous_steps(self, variables):
        # Iterate on completed steps
        if os.path.exists(os.path.join(self.scope_folder, '..', 'completed')):
            with open(os.path.join(self.scope_folder, '..', 'completed'), 'r') as completed:
                for step_name in completed:
                    step_name = step_name.strip()
                    self._load_data_from_scope(variables, Scope(self.cache_folder, self.seed, self.id[:-1] + [step_name]))

    def persist(self, variables):

        var_folder = os.path.join(self.scope_folder, 'vars')
        if not os.path.exists(var_folder):
            os.makedirs(var_folder)
        
        tmp_folder = os.path.join(self.scope_folder, 'tmp')
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        for name in self.modified_variables:
            variable = variables[name]
            if len(variable.scope.id) > len(self.id):
                # Not supposed to happen, raise an error in future versions.
                continue
            if variable.scope.id == self.id:
                object_path = os.path.join(var_folder, name)
            else:
                object_path = os.path.join(tmp_folder, name)

            if type(variable.value).__module__ == np.__name__:
                np.save(object_path + '.npy', variable.value)
            else:
                with open(object_path + '.pkl', 'wb') as f:
                    pickle.dump(variable.value, f)

        # Mark as completed
        with open(os.path.join(self.scope_folder, '..', 'completed'), 'a') as f:
            f.write(self.id[-1] + '\n')
    
    def archive(self):
        shutil.rmtree(os.path.join(self.scope_folder, 'tmp'))



class Experiment():

    def __init__(self, db, seed, folder='./cache', verbose=0):
        self.db = db
        self.seed = seed
        self.folder = folder
        self.verbose = verbose

        # Open the database connection
        self._db_conn = dataset.connect(self.db)

        self.scope = Scope(folder, seed, ['main'])
        self.variables = dict()
    
    def _log(self, verbosity, message):
        if self.verbose >= verbosity:
            print(message)

    def variable(self, key, default):
        if not key in self.variables:
            self.variables[key] = LazyValue(self.scope, key, value=default)
        else:
            if len(self.variables[key].scope.id) == len(self.scope.id):
                self.variables[key].scope.id = self.scope.id
        return Variable(self, key)

    def _get_variable(self, key):
        if not key in self.variables:
            raise ValueError('Unknown variable {}'.format(key))
        return self.variables[key].get_value()

    def _set_variable(self, key, value):
        if not key in self.variables:
            raise ValueError('Unknown variable {}'.format(key))
        self.variables[key].value = value
        self.scope.tag_modified_variable(key)
    
    def log_value(self, config, key, value, **kwargs):
        table = self._db_conn[key]
        table.upsert(dict(value=value, **config, **kwargs), config.keys())


    def iter(self, step_name, items, op_name, cache=True, force=False):

        parent_scope = self.scope

        resuming = False
        sentinel = object()

        for prev_item, curr_item in _paired_iter(itertools.chain([sentinel], items, [sentinel])):

            # Iterate until the current iteration is not computed
            curr_scope = Scope(self.folder, self.seed, parent_scope.id + ['{}_{}'.format(step_name, str(curr_item)), op_name])
            self.scope = curr_scope

            if curr_scope.is_completed() and cache and not force:
                self._log(1, 'Context {} {} {} already completed'.format(step_name, op_name, curr_item))
                resuming = True  # This indicates that we further need to load the previous context.
                continue

            # Set information about the previous context
            prev_scope = None
            if prev_item is not sentinel:
                prev_scope = Scope(self.folder, self.seed, parent_scope.id + ['{}_{}'.format(step_name, str(prev_item)), op_name])

            if resuming:
                prev_scope.load_data(self.variables)
                resuming = False

            if curr_item is sentinel:
                # We have reached the last iteration, we can stop
                break

            # Now we load data from a possible previous step
            curr_scope.load_previous_steps(self.variables)

            yield curr_item

            if not cache:
                continue

            # Computation is successful. We store the current context on disk.
            curr_scope.persist(self.variables)
            
            # The previous context is now useless, we remove all temp variables from it that are not from its context
            if prev_scope:
                prev_scope.archive()

        # Remove unnecessary variables from context
        for name in self.variables:
            if self.variables[name].scope == self.scope.id:
                del self.variables[name]
    
        self.scope = parent_scope
